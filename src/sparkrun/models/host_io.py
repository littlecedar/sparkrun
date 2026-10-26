"""Owned host sessions and cooperative model-cache publication leases."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import PurePosixPath
import uuid
import threading

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest
from sparkrun.models.manifest import STATE_DIRECTORY
from sparkrun.models._publication import HeldThroughAlias, PublicationLease
from sparkrun.models.observation import observation_script, parse_observations, path_assignment
from sparkrun.transports.session import HostCommandResult, HostSession, HostSessionError, SshHostSession, StreamingHostSession


class ModelHostIO:
    def __init__(self, ssh_kwargs: dict | None = None, *, session: HostSession | None = None):
        kw = ssh_kwargs or {}
        self.session = session or SshHostSession(**{key: kw[key] for key in ("ssh_user", "ssh_key", "ssh_options") if key in kw})
        self.local = SshHostSession()
        self.owns_session = session is None
        self._stack = ExitStack()
        self._operation = uuid.uuid4().hex
        self._locks: dict[tuple[str | None, str], str] = {}
        self._owners: dict[str, PublicationLease] = {}
        self._mutex = threading.RLock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if exc[0] is None:
                self.check_leases()
        finally:
            try:
                self._stack.close()
            finally:
                self.local.close()
                if self.owns_session:
                    self.session.close()

    def check_leases(self):
        with self._mutex:
            owners = tuple(self._owners.values())
        for owner in owners:
            owner.check()

    def cancelled(self):
        try:
            self.check_leases()
        except ModelArtifactError:
            return True
        return False

    def release_leases(self):
        """End a preparation phase before acquiring a larger, ordered lock set.

        Callers must re-observe after reacquisition. A lost lease is terminal,
        even here; releasing it must never turn failure into a successful retry.
        """
        with self._mutex:
            self.check_leases()
            self._stack.close()
            self._owners.clear()
            self._locks.clear()

    def execute(
        self,
        host: str | None,
        script: str,
        *,
        timeout: float | None = 30,
        cancelled=None,
        stdout=None,
        idle_timeout: float | None = None,
        watch=None,
    ) -> HostCommandResult:
        self.check_leases()
        session = self.local if host is None else self.session

        def interrupted():
            self.check_leases()
            return cancelled is not None and cancelled()

        try:
            if cancelled is not None or self._owners or watch is not None or timeout is None:
                from sparkrun.models._stream import execute_stream

                result = execute_stream(
                    session,
                    host or "localhost",
                    script,
                    timeout=timeout,
                    cancelled=interrupted,
                    stdout=stdout,
                    # Lease supervision must not truncate previously unbounded
                    # metadata reads. Bulk copy output retains its existing cap.
                    stdout_limit=None if cancelled is None and stdout is None else 16 * 1024**2,
                    idle_timeout=idle_timeout,
                    watch=watch,
                )
            else:
                result = session.execute(host or "localhost", ["bash", "-s"], input_data=script.encode(), timeout=timeout)
            self.check_leases()
            return result
        except (HostSessionError, OSError) as exc:
            raise ModelArtifactError("model host session unavailable on " + (host or "control")) from exc

    def publish(self, host: str | None, cache: str, repo: str, path: str, contents: str):
        """Commit metadata through the process holding this repository's flock.

        Aliases may name the shared mount differently; translate to the owner's
        root before sending the write. Loss never silently reacquires a lease.
        """
        nonce = self.lock(host, cache, repo)
        owner = self._owners[nonce]
        relative = PurePosixPath(path).relative_to(cache)
        if ".." in relative.parts:
            raise ModelArtifactError("model publication escapes its cache")
        owner.publish(str(PurePosixPath(owner.root) / relative), contents, check=self.check_leases)

    def checked(self, host: str | None, script: str, *, timeout: float = 30) -> str:
        result = self.execute(host, script, timeout=timeout)
        if result.returncode:
            raise ModelArtifactError(
                "model operation on %s failed (rc=%d): %s"
                % (host or "control", result.returncode, result.stderr.decode(errors="replace")[-1500:])
            )
        return result.stdout.decode()

    def root(self, host: str | None, cache: str) -> str:
        text = self.checked(host, path_assignment("root", cache) + 'realpath -m -- "$root"\n')
        # realpath prints one terminating newline; keep newlines inside paths.
        path = text.removesuffix("\n")
        if not PurePosixPath(path).is_absolute() or "\x00" in path:
            raise ModelArtifactError("host returned an invalid model cache path")
        return path

    def observe(
        self, host: str | None, manifest: ModelArtifactManifest, cache: str, *, checksums: bool = False, timeout=None, cancelled=None
    ):
        result = self.execute(
            host,
            observation_script(manifest, str(PurePosixPath(cache) / manifest.relative_snapshot), checksums=checksums),
            timeout=timeout if timeout is not None else (1800 if checksums else 60),
            **({"cancelled": cancelled} if cancelled is not None else {}),
        )
        return parse_observations(
            host or "control", manifest, result.stdout.decode(errors="replace"), returncode=result.returncode, checksums=checksums
        )

    def read(self, host: str | None, path: str) -> str | None:
        result = self.execute(host, path_assignment("p", path) + 'if [ ! -e "$p" ]; then exit 3; fi\ncat -- "$p"\n')
        if result.returncode == 3:
            return None
        if result.returncode:
            raise ModelArtifactError("could not read model manifest on " + (host or "control"))
        return result.stdout.decode()

    def lock(self, host: str | None, cache: str, repo: str) -> str:
        """One lease per shared repo write domain, held across provider callbacks.

        The nonce in the flock file detects a lease we already hold through
        another host/mount alias. Never unlink flock files: that would allow
        two writers to lock different inodes under the same pathname.
        """
        with self._mutex:
            self.check_leases()
            path = str(PurePosixPath(cache) / STATE_DIRECTORY / (repo.replace("/", "--") + ".lock"))
            key = host, path
            if key in self._locks:
                return self._locks[key]
            existing = self.read(host, path)
            if existing:
                existing = existing.splitlines()[0]
            if existing is not None and existing in self._owners:
                self.check_leases()
                self._locks[key] = existing
                return existing
            session = self.local if host is None else self.session
            if not isinstance(session, StreamingHostSession):
                raise ModelArtifactError("model publication requires a host session with supervised streaming leases")
            from sparkrun.core.model_distribution import lease_timeout

            nonce = self._operation + ":" + uuid.uuid4().hex
            try:
                owner = PublicationLease(
                    session,
                    host,
                    cache,
                    path,
                    nonce,
                    check=self.check_leases,
                    owned=self._owners.__contains__,
                    acquire_timeout=lease_timeout(),
                    timeout_setting="model_cache_lease_timeout_s",
                )
            except HeldThroughAlias as alias:
                # The read above saw the lock before our holder's nonce was
                # visible through this name (NFS attribute caching).
                self.check_leases()
                self._locks[key] = alias.nonce
                return alias.nonce
            self._stack.callback(owner.process.close)
            self._owners[nonce] = owner
            self._locks[key] = nonce
            self.check_leases()
            return nonce
