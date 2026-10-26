# Model cache validation and distribution

Manifest preparation binds a model's endpoint, immutable commit, selected files,
checksums and runtime paths together. Core validates every required host before
launch; a download, rsync exit code or plugin success receipt is insufficient.

## Default behavior and policy

No configuration change is needed to adopt manifests. The default `legacy`
policy uses metadata validation with compatibility fallback. Optional settings
in `config.yaml`:

```yaml
model_cache_validation: legacy        # legacy (default) | metadata | checksum
model_file_selection: auto            # auto | all | safetensors
model_distribution_provider: auto     # auto | builtin | registered provider name
model_transfer_streams_per_host: 2     # concurrent rsync processes per destination
model_transfer_max_streams: 16         # total rsync processes per copy operation
model_transfer_idle_timeout_s: 600     # cancel a copy/download whose bytes stop advancing
model_transfer_timeout_s: 0            # optional total ceiling; 0 (default) = none
model_transfer_stall_retries: 1        # resume a stalled rsync this many times
model_cache_lease_timeout_s: 86400     # wait this long for a busy cache lease; 0 = no limit
# model_distribution_fallback: true   # optional override of provider policy
```

Both `legacy` and `metadata` resolve authoritative inventories, select required
files, check existence/type/readability/size, and persist manifests automatically.
Manifests are stored in `.sparkrun-model-artifacts/` under the HF cache home on
the inventory source and launch targets. Lookups cover both the requested ref and
resolved commit, so a later pinned recipe can reuse an inventory learned from an
unpinned launch. Inventory is saved before acquisition/repair, so even an
interrupted or failed preparation retains the expected file list for a later
offline run. Stored manifests describe expected files; they never replace
checking the current files or certify that a download completed.

The difference is what happens when inventory is unavailable:

- `legacy` uses saved inventory if available, warning when an online mutable-ref
  refresh fails. Without inventory it warns that the cache is unverified and uses
  the previous cache/download checks. This lets existing offline caches keep
  working and migrate automatically on a later successful online preparation.
- `metadata` requires authoritative inventory; missing inventory or a failed
  online mutable-ref refresh fails preparation.
- `checksum` has the same inventory requirement and additionally reads selected
  payloads to check authoritative SHA-256 or Git blob SHA-1 hashes. Missing
  digests are unverified, never successful checksum checks.

Every policy rejects invalid manifests and requires repair or failure for known
missing/damaged files. Repair, transport and workload access failures never
downgrade to the old checks. Selecting a model provider requires the strict
inventory contract even if the configured validation policy is `legacy`.
Explicit per-model file filters also require inventory in every policy: a saved
manifest for the same selection may be reused, but the unfiltered compatibility
fallback is disabled.

Named clusters can override those settings without changing global configuration:

```yaml
distribution:
  model:
    validation: metadata
    file_selection: auto
    provider: builtin
    streams_per_host: 2
    max_streams: 16
    idle_timeout_s: 600
    timeout_s: 0
    stall_retries: 1
    # fallback: false
```

Existing `enabled`, `preserve_perms` and `skip_fan_out` remain compatible with the
compatibility fallback. Manifest preparation reobserves each host even on shared storage;
already-complete destinations need no copy. It does not recursively chown caches.
Its snapshot copies preserve times when supported, without preserving ownership
or permissions. Permission failures fail preparation rather than triggering a
large redownload under a different identity.

## Select files without redundant weights

Recipe entries in `distribution_config.models.entries` and `additional_models`
can specify `allow_patterns` and `ignore_patterns` to stage an explicit subset,
including ancillary files without weights. See the [recipe example](../RECIPES.md#download-selected-files-from-an-auxiliary-model).
These filters replace automatic selection for that entry, apply to acquisition
and every transfer leg, and do not implicitly fetch weight indexes or shards.
An empty selection fails. Unselected files already in a snapshot remain untouched.

Filtered requests have distinct lookup keys and manifest identities. A filtered
manifest cannot validate a full request or a different selection. The same
selection can be reused offline, including at its resolved commit. Partial
preparation does not publish HF branch refs such as `refs/main`; use its validated
snapshot path. A full request still checks its own expected inventory, even when
some files were staged by a previous filtered request.

The Python `ModelArtifactRequest` and daemon model-cache resolve, inspect,
import, and repair operations accept the same `allow_patterns` / `ignore_patterns`
fields. Pass the same patterns when inspecting or repairing a filtered cache.

With inventory available, the default `auto` selection reads the root
`model.safetensors.index.json` at the resolved commit and
includes every shard named by its `weight_map`. It excludes unreferenced root
checkpoints and recognized `original/` and `metal/` alternative weight payloads.
Configuration, tokenizer, license and other ancillary files remain included.
Unfamiliar nested component trees remain included to avoid discarding separate
vision/encoder weights. No shard count is inferred from safetensors filenames.
Without a root index, the complete repository inventory is selected.

Index-based selection accepts at most **128 MiB for the index JSON itself**;
this is an inventory parsing bound, not a limit on weight files, total model
size, or VRAM. Quantized MoE exports can list each expert's weights and scales
separately; Ornith-1.5-397B NVFP4, for example, has a 37.62 MiB index with
369,891 tensor mappings. Missing index-size metadata is reported separately from an
oversized index, with the repository, resolved revision, and endpoint. The
same checks apply to controller and delegated acquisition. Selection `all`
skips index-based filtering and can include additional weight formats; it does
not repair missing inventory metadata.

This addresses [#142](https://github.com/spark-arena/sparkrun/issues/142) without
checking for a GPT-OSS model name. At GPT-OSS-120B commit
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, metadata and the index select 15 root
shards: about **65.28 GB instead of 195.76 GB**, avoiding about 130.49 GB of
alternative weights. This measurement uses inventory sizes, not a weight download.

Use `all` for a loader requiring additional weight formats; `safetensors` requires
a root index and fails if none exists. GGUF uses its quantization selection,
checks multipart completeness, and binds its selected weight and projector from
that same revision. It never picks an arbitrary cached snapshot at launch.

## Inspect, migrate and repair

These commands print per-host tables; pass `--json` for the structured reports.
Inspection/import return nonzero when a target is incomplete or unverified.
Targets resolve like every other cluster command: `--cluster`, `--hosts` or
`--hosts-file`, else the default cluster, else `default_hosts` from config, else
localhost. The cluster's SSH user applies. `--cache-dir` is the target's HF home, not
`hub/`; without it the cluster's `cache_dir` is used, else the head node's HF
home is probed the way `run` does. `resolve` runs on the control machine only.

```sh
# Metadata only, offline and read-only; no recursive cache traversal.
sparkrun model-cache inspect org/model --revision COMMIT --cache-dir /models/hf

# Resolve authoritative inventory online (plus an index, at most 128 MiB).
sparkrun model-cache resolve org/model --revision COMMIT --output manifest.json

# Import reviewed inventory for an existing/offline cache. Import is not repair.
sparkrun model-cache import org/model --revision COMMIT --manifest manifest.json

# Without a manifest, resolve origin metadata and confirm before saving it.
sparkrun model-cache import org/model --revision COMMIT

# Explicit consent for unattended origin resolution; stdout stays JSON.
sparkrun model-cache import org/model --revision COMMIT --yes --json

# Require a supplied manifest and prohibit origin resolution.
sparkrun model-cache import org/model --revision COMMIT --manifest manifest.json --offline

# Repair exactly the missing/wrong-size files and revalidate each target.
sparkrun model-cache repair org/model --revision COMMIT --hosts node1,node2

# Explicit checksum audit/repair, including same-size corruption.
sparkrun model-cache inspect org/model --revision COMMIT --checksum
sparkrun model-cache repair org/model --revision COMMIT --checksum
```

`import --manifest` is preferred for reviewed, reproducible inventories and
always works without contacting the model origin. When `--manifest` is omitted,
`import` resolves metadata once on the control machine and uses the same
immutable commit and selected inventory on every target host. Resolution may
download a weight index (at most 128 MiB), but never model weights. Before saving,
the interactive command displays the origin URL, repository, full commit SHA,
file count and target hosts, then asks for confirmation. Declining leaves target
manifests unchanged. `--yes` permits the import without prompting; unattended and
`--json` imports require it when no manifest is supplied. The resolved identity
is printed to stderr so stdout remains machine-readable with `--json`.

`--offline` requires `--manifest`, even with `--yes`. An invalid supplied manifest
fails without falling back to the origin. Unavailable metadata or authentication
failures also stop the import; local directory listings never substitute for an
expected inventory. Import saves a valid manifest even when the cache is
incomplete, reports the missing or invalid files, and exits nonzero. Use `repair`
to acquire missing files, or `--checksum` to hash existing files during import.
Import does not refresh usage or enroll snapshots in retention; use `adopt` to
start tracking an existing download.

Specify `--revision` when importing an existing download. If it is omitted, the
origin resolves its current default revision, which may differ from an older
cached snapshot. Origin-resolved manifests use the same validation and provenance
as `resolve`; confirmation accepts that source and revision, rather than marking
the cache valid or granting permission to execute model code.

Pass the same `--file-selection` and GGUF `--projector` used for preparation when
inspecting or importing. `repair` accepts `--transfer-mode`, `--local-cache-dir`,
`--offline` and `--dry-run`. Offline preparation uses saved/imported inventory and
existing controller/cluster files; it never calls the Hub or installs tools. A
manifest-less offline cache is **unverified**, not implicitly valid or corrupt;
only `legacy` launch/distribution may proceed using the previous presence checks.
Dry-run makes no metadata requests, probes or cache writes. Origin resolution
for mutable refs is online even when the payload is already cached. A pinned
commit with saved inventory needs no origin request.

The Python surface is `sparkrun.api.model_cache`: `resolve_manifest`,
`inspect_model`, `import_manifest`, `import_model`, and `repair_model`, with typed
requests, manifests, reports and prepared bindings. `import_manifest` remains an
offline operation requiring a manifest. `import_model` accepts a supplied
manifest or requires explicit `fetch_manifest=True` for origin resolution;
`offline=True` still requires a supplied manifest. Its optional
`confirm_manifest(manifest) -> bool` callback can review the resolved inventory
before any target cache is opened; returning false cancels the import. Explicit
`token` credentials override local HF configuration. `api.materialize()` remains
read-only.

For a pre-placed directory, put the trusted manifest at
`<directory>/.sparkrun-model-manifest.json`, with inventory paths relative to that
directory. For a pre-placed GGUF file, use
`<file>.sparkrun-model-manifest.json`, paths relative to its parent, and set
`weight_path` to that filename. These inventories are checked before launch,
including under the default policy; missing inventories produce an explicit unverified
migration warning. Core does not acquire or repair pre-placed paths automatically.
Without a manifest, local HF directories used by native vLLM, SGLang, Atlas and MAX still undergo
basic config, index, shard and readability checks on every target. GGUF files use separate checks
and need no `config.json`. Docker and local executors check the selected files through the workload's
mounts and execution identity before replacement; images need no Python for that access check.
These checks do not create a manifest, establish cross-node content identity, or validate weight hashes.

## Usage, retention and LRU cleanup

Model preparation records usage beside each cache in
`HF_HOME/.sparkrun-model-artifacts/usage/`. These mutable records are separate
from the immutable file manifests. They track preparation, acquisition,
distribution-source use, launch attempts and confirmed workload release, keyed
by repository and immutable revision. Inspection and manifest import do not
refresh usage or enroll unknown files in cleanup. Preparation needs a writable
Sparkrun metadata directory, including for a warm cache.

The records belong to the cache host, not the controller's daemon database.
Delegated downloads and fan-out therefore record usage on the head and workers
without storing model weights on the controller. Shared cache mounts share the
same usage records and claims.

`list`, `prune`, `pin`, `reconcile` and `delete` work per **cache location**, not
per host: hosts that see the same cache (one NFS/SMB export, or one machine
reached under two names) get a single report naming all of them, and the action
runs once. Identity is probed read-only: mount source, the path within the
export and its inode for a network filesystem; boot id, device and inode for a
local one, so two machines' local disks are never merged. Before a mutating
action, sharing is also confirmed by writing a nonce through one host's path and
reading it back through each other's; a host that cannot confirm it is handled
separately. Byte totals across reports can therefore be added up.

```sh
# Read-only inventory, including untracked snapshots and protection reasons.
sparkrun model-cache list --cluster lab
sparkrun model-cache list --hosts node1,node2 --cache-dir /models/hf --json

# Preview LRU cleanup (the default); a byte budget applies per cache.
sparkrun model-cache prune --hosts node1 --cache-dir /models/hf \
  --min-age-days 7 --max-bytes 500G

# Explicitly apply the same policy. Nothing deletes automatically by default.
sparkrun model-cache prune --hosts node1 --cache-dir /models/hf \
  --min-age-days 7 --max-bytes 500G --apply

# Keep a revision; use --remove to remove its retention pin. A ref such as
# main resolves through each cache's own refs/, so caches may pin different commits.
sparkrun model-cache pin org/model --revision main \
  --hosts node1 --cache-dir /models/hf
```

`prune`, `adopt` and `reconcile` preview by default and change nothing without
`--apply`; `prune --apply` does not ask again, so review the preview first.
`delete` always previews, then asks for confirmation (`--yes` skips it,
`--dry-run` stops after the preview). `repair` only adds files and runs unless
given `--dry-run`.

`prune` defaults to a dry run and a seven-day minimum age. Without a byte
target it selects all eligible revisions older than that age. `--max-bytes`
budgets private allocations in tracked snapshots, including blobs only that
snapshot links to, and excluding blobs other revisions share and untracked data. `--min-free-bytes` instead targets actual free
space on the cache filesystem. With both targets, cleanup continues until
both are met or no eligible revision remains. The CLI exits nonzero if a
target cannot be met; it never overrides protections to reach a target.

Reports separate logical file sizes, reclaimable snapshot allocations, planned
removals and filesystem free space. Allocation counts exclude symbolic links and
count a hard-linked file once only when every name for its inode is being removed,
including snapshot, repository-blob and shared-store names. A surviving hard link
prevents crediting that allocation. These are estimates of disk reclamation, not a
promise about filesystem compression, reflinks or concurrent disk activity;
`free_bytes_after` is measured again when applying cleanup. Dry-run free-space
projections use the allocation estimates. Evicting a snapshot in the Hub's layout (entries are symlinks into `blobs/`)
also removes the blobs **only that snapshot** links to, so its space is actually
freed; a blob any other revision of the model still links to is kept, and
prune refuses the revision while a Hugging Face download of the model is active.
Shared-store payloads go through huggingface_hub's own lock and `.refs` protocol,
as with `delete`.

Only tracked revisions with a known file inventory are eligible. Unknown
snapshots, unexpected files, pinned revisions, busy publication locks, invalid
metadata and active workload claims are protected. Arbitrary pre-placed model
directories are outside the cleanup scope. To enroll an existing HF snapshot,
prepare/repair it successfully; importing a manifest alone does not grant
cleanup ownership. Select `--file-selection all` if all repository files need
to be included in that preparation.

Claims are written before preparation releases its repository locks and remain
after the launching CLI exits. All prepared models, including draft models and
projectors in their selected inventory, are covered. A manifest-less legacy
launch conservatively protects every previously tracked revision of its model.
Confirmed `stop`/`stop-all` releases the corresponding claims and restarts the
retention age. A failed or interrupted activation retains its claims; neither
age nor a dead controller proves that remote workloads stopped.

A relaunch under the same workload id (an `export systemd` unit restarting, or
`run` again) keeps the earlier launch's claims in the job record. At the
replacement barrier it releases them on every host where its status sweep, through
the executor the earlier launch recorded, saw that workload not running; where it
still runs, its containers are replaced by name without confirmation and the
claim is kept for the next stop. `stop` also tears the id down on hosts that only
an earlier launch claimed (the scheduler may have placed the relaunch elsewhere),
which is a no-op where nothing runs. If a claim still cannot be released, `stop`
succeeds with a warning naming `model-cache reconcile` and keeps the job record.

Retry a normal stop to reconcile a known workload. Otherwise let sparkrun find
the claims whose workload is gone (default cluster, or `--cluster`/`--hosts`):

```sh
sparkrun model-cache reconcile --all            # preview, every model
sparkrun model-cache reconcile --all --apply    # release
sparkrun model-cache reconcile org/model        # one model only
```

It sweeps the targets' status once and releases a claim only when every host in
the claim was observed, its workload runs on none of them, no pending operation
on this control machine names it, and it is older than `--min-age-minutes`
(default 60) plus five minutes of allowance for clock skew: `created` is the
clock of whichever control machine launched it, so even `--min-age-minutes 0`
keeps a claim made in the last five minutes. Claims of another distribution's
workloads (status cannot see them) are kept. A claim records the executor its
launch used, and is kept unless the sweep queried that executor. A claim from an
older sparkrun records none, and is kept while any executor on the cluster's
substrate is turned off by a feature flag on this machine (another controller may
have launched it through `local`, leaving no job record here); enable the flag
named in the reason, for this run if need be (`SPARKRUN_FEATURE_EXECUTOR_LOCAL=1`),
so the sweep covers it. The age floor covers a launch that has claimed its model but not
yet started containers, possibly from another control machine; lower it only
when nothing else is launching. Claims on unreachable hosts, or naming hosts
outside the targets, are reported and kept.

To release one claim by hand (`--claim`), inspect **every host listed in the outstanding
claim**, stop the workload and any in-flight launch, then release the exact
claim shown by `list`:

```sh
sparkrun model-cache reconcile org/model --revision FULL_COMMIT_OR_REF \
  --claim CLAIM_ID --hosts node1 --cache-dir /models/hf --confirm-inactive
```

Reconciliation removes a protection record, not model files, and refreshes its
retention age. For a single claim the confirmation is an operator assertion;
that form does not infer process inactivity. External applications and older Sparkrun
processes do not participate in these claims/locks: pin revisions they use,
or leave automatic cleanup disabled on their shared caches.

Optional automatic cleanup runs before manifest-based model preparation:

```yaml
model_cache_prune:
  enabled: false                  # opt in explicitly
  min_age_days: 7
  min_free_bytes: 100G             # free-space target per cache
  # max_bytes: 500G               # optional private snapshot budget (must be positive)
```

Enabling it requires at least one byte target. Sizes take a plain byte count or
a suffix: `K`/`M`/`G`/`T` (or `KiB`...`TiB`) are binary, `KB`/`MB`/`GB`/`TB`
decimal, in any letter case. `prune --max-bytes` / `--min-free-bytes` take the
same forms. It reuses the explicit prune
implementation, excludes the repository being prepared, and retains pins and
workload claims. It warns if protected content prevents meeting the target;
this is not a guarantee that a subsequent download will fit. It does not run
on dry-run launches, and no background daemon is required. Cluster-seed
retention can currently be implemented by pinning the revision on the desired
head host; there is no automatic cluster-wide replica placement policy.

Inventory and cleanup use Python 3 and standard filesystem tools on Linux cache
hosts, through the existing SSH/host session. They never contact the model
origin or install tools. Existing metadata inspection remains independent of
this Python helper. CLI and daemon callers use the same Python API:
`inventory_cache`, `prune_cache`, `pin_revision`, `reconcile_claim` and `adopt_cache`
(`reconcile_claims`, behind `reconcile --all`, is Python/CLI only).
The daemon exposes `model_cache.list`, `model_cache.prune`, `model_cache.pin`,
`model_cache.reconcile` and `model_cache.adopt` (and matching `/v1/model_cache/...`
routes). Prune, reconciliation and adoption require admin scope; prune and adopt
default to dry-run over the API as well.

## Adopting existing downloads

A revision downloaded outside sparkrun (`hf download`, a notebook, an older sparkrun)
has no usage record: `list` hides it and `prune` never frees it. `adopt` writes that
record, and only that: nothing is downloaded, moved, deleted, or written as a manifest.

```sh
sparkrun model-cache adopt org/model --cluster lab              # preview
sparkrun model-cache adopt org/model --cluster lab --apply
sparkrun model-cache adopt --all --cluster lab --offline         # every model, no origin requests
```

Name a model (optionally `--revision`, a commit or a ref such as `main`) or pass `--all`;
one is required, so a single command cannot enroll everything other tools on the host
use. Each snapshot is judged before it is adopted:

- **Online** (default): every file present must exist in the model origin's inventory
  for that commit, at the same size. Missing files are fine (a partial download, such as
  one GGUF quant, is normal); a file the origin does not know, or knows at another size,
  refuses adoption.
- **`--offline`**: every entry must be a link into the blob store, the layout the Hub
  client writes and names by content. A plain file proves nothing offline and refuses.

`last_used` starts *now*, so nothing becomes prune-eligible the moment it is adopted;
`--last-used mtime` uses the snapshot's newest file instead, which can make an old
download eligible at once. `--pin` adopts and protects in one step. Once adopted, a
Hub-layout revision is pruned with the blobs only it uses.

Adopted revisions age by **sparkrun's** use only: use by other tools (notebooks, `hf`,
containers started outside sparkrun) never refreshes `last_used`. With automatic pruning
enabled, a revision other tools still rely on therefore becomes a cleanup candidate once
the retention window passes; pin those. The origin an adoption records is replaced by the
first launch that prepares the revision, so adopting with a different spelling of the
endpoint (or `--offline`) never makes a later launch fail on an origin mismatch.

## Explicit deletion

`delete` removes a model, or one revision of it, from the target hosts' caches.
Naming the model is the authorization, so pins, retention age and untracked
status do not protect it. It previews what would be removed, then asks for
confirmation (`--yes` skips the prompt; off a terminal it is required).

```sh
sparkrun model-cache delete org/model --cluster lab --dry-run
sparkrun model-cache delete org/model --revision FULL_COMMIT_OR_REF --cluster lab
```

Python `api.model_cache.delete_model` and daemon `model_cache.delete` clients
can pass `allowed_revisions` from their confirmed preview when applying with
`dry_run=False`. An empty list permits no revisions; omitting the field authorizes
the model/revision selection at execution time.

- A revision with an outstanding workload claim blocks the delete on that host
  unless `--force`; its weights are probably in use by a running workload.
- An active Hugging Face download (a held lock under `hub/.locks`) or a busy
  publication lock always refuses, in the preview as well as when applying.
- If any host would refuse, nothing is deleted on any host. Only the revisions
  shown in the confirmed preview are deleted; one that appeared since refuses.
- A revision delete removes its snapshot, refs pointing at it, its usage record,
  and per-model blobs no other snapshot of that model references. A whole-model
  delete removes the repository directory and its usage records.
- Shared-store blobs (huggingface_hub's cache-wide `hub/blobs`) follow
  huggingface_hub's own protocol: the per-blob `<hash>.lock` that downloads take
  when they publish or link content, and the `<hash>.refs` manifest. A blob goes
  only when its manifest shows no other live reference. A missing or unreadable
  manifest, a lock held for more than ten seconds, or an unknown store layout
  keeps it. Nothing outside the cache is removed, even through a symlink.
- Revisions often share most of their files. Deleting one revision of a model
  whose other cached revision uses the same weights frees only what differs;
  the preview says how much stays and why.
- Saved manifests are kept, so `model-cache repair` can restore the model.
- Links into the cache from outside HF_HOME (for example old `local_dir`
  checkouts that symlink cache blobs) are invisible to this, as they are to
  `hf cache rm`.
- Compared with `hf cache rm`, this also refuses while a download of the same
  repository is running and honors sparkrun's claims and publication lock, and
  it needs no `huggingface_hub` on the hosts.

## Model disk-space checks

Manifest preparation checks the source and every destination before starting
payload downloads or copies, after reusing any files already cached on each
host. It compares the sizes of the remaining selected files with the available
space on the model cache's filesystem. Shared-cache aliases count once;
separate caches on the same host filesystem share a capacity budget.

An insufficient-space error identifies the host and cache path and reports
required bytes, available bytes, and the shortfall. No payload transfer starts
when this preflight fails. Saved inventory and completed files remain available
for a retry after space is freed. Capacity is checked again before acquisition
and each copy leg, because a source download or another process can consume
space after the initial check.

The budget includes the full size of missing or damaged files: replacement
keeps the old entry until the new file is ready. Existing partial copies are
not credited, since transfer providers may stage a fresh copy. The check does
not reserve space or predict filesystem quotas; disk-full and quota failures
during a transfer still stop the operation with the affected host identified.
Legacy transfers without an authoritative inventory retain their existing
behavior.

## Parallel model copies

The builtin manifest transport copies to multiple destinations concurrently.
Each destination can use two rsync processes by default, with sixteen processes
in total per copy operation (one source). Set `model_transfer_streams_per_host`
and `model_transfer_max_streams` in global configuration, or `streams_per_host`
and `max_streams` under a cluster's `distribution.model`. Values must be positive
integers. Set both to `1` for serial transfers. These limits apply to the builtin
manifest copy path; plugins and legacy inventory-unavailable fallback retain
their own transfer behavior. Separate operations have separate limits.

Only files that fail the selected validation policy enter the queue. Files at
least 64 MiB run as individual jobs; smaller files are grouped into batches up
to 64 MiB. Admission rotates across destinations, with larger files first.
Multiple streams use independent SSH connections over the selected transfer
addresses, retaining SSH authentication and host-key verification. Rsync
compression is off; bulk SSH connections disable multiplexing and default
to no compression.

Shared-cache aliases identified by the held publication leases receive one
copy. Jobs use disjoint file lists, retain rsync partial files for retries, and
replace snapshot files atomically. Cancellation and deadlines stop owned copy
processes without closing a borrowed session or releasing publication leases
before workers finish. Core revalidates every target before publishing runtime
bindings, including aliases. A single large file stays a single transfer;
this scheduler does not split GGUF files into byte ranges.

After a source or target passes validation, preparation removes its
`.sparkrun-partial` copies of the verified files under the repository lease.
This also clears leftovers when a later run finds the snapshot already complete
or restores it from cached blobs without transferring those files. Incomplete
snapshots retain their resume data; partials for files outside the selected
manifest are preserved for their own unfinished transfers.

### Stall detection, not a deadline

A transfer may take as long as it needs while it is moving. Each rsync is
watched through its `--info=progress2` byte count: when the count has not grown
for `model_transfer_idle_timeout_s` (default 600 seconds), that rsync is
cancelled and resumed from its partial file, up to `model_transfer_stall_retries`
times (default 1). After that the host fails with the file, the idle time and
the bytes moved, for example `model transfer stalled: no progress for 600s
(model-00012-of-00040.safetensors, 3.10 of 4.88 GiB transferred)`. Repeated
progress output with an unchanged count does not count as progress. The idle
clock starts with the process, so a connection that hangs before any data moves
is caught too. Two windows move no bytes while healthy and are not watched:
after the last byte (the receiver closing and flushing on a slow shared cache)
and, on a resumed attempt, before the first byte (rsync checksumming the
partial file). Rsync's own `--timeout`, set to the same interval, bounds those
and makes the remote end exit if the controller is gone. The Hugging Face repair
download is watched the same way through its sampled byte count, with the idle
clock restarted once the download adapter has bootstrapped. This applies both to
controller downloads in local/push mode and to downloads on a delegated host. A
controller download stops on timeout or publication-lease loss and removes its
temporary staging files, preserving completed snapshot files. The legacy fallback
path uses rsync `--timeout` alone, at a fixed 600 seconds.
`SPARKRUN_NO_RSYNC_RETRY=1` also disables stall resumes.

There is no total ceiling by default: a fixed one failed every transfer whose
size outgrew the link, however healthy it was. Set `model_transfer_timeout_s`
(or `timeout_s` under a cluster's `distribution.model`) to impose one.

CLI and API feedback report aggregate/per-host logical bytes processed,
active stream counts, and logical bytes per second. Counters include data
reused by rsync and remain monotonic across retries; they do not measure bytes
sent on the network. Completion of a copy is followed by core validation.
Tune stream counts against actual disk and network throughput; additional
streams cannot exceed the source NIC or storage bandwidth.

## Hugging Face transports

Weight downloads prefer Xet when the repository supports it and `hf_xet` is
available; the pinned Hub dependency installs it on supported architectures.
Xet handles chunk-based transfers, and cache-backed downloads can share blobs
across repositories. Snapshot repair uses a private staging directory and
still verifies the selected files before launch.

Hub uses HTTP when Xet is unavailable, when the file is not Xet-backed, or when
`HF_HUB_DISABLE_XET=1` is set in the downloading process. It does not silently
retry every failed Xet transfer over HTTP: authentication, access, integrity,
and transfer failures remain failures. Set the override on the controller or
remote downloading host as appropriate. An offline run does not try either
network transport.

Advisory metadata reads deliberately use bounded HTTP instead of Xet's separate
Rust transport. The overall metadata deadline remains in force even when the
HTTP library retries a request. Hub 2.x uses `httpx2`; Sparkrun takes its timeout
and exception types from `huggingface_hub.utils.httpx` and preserves Hub's client
factory, redirect policy, and event hooks. HTTPS uses the OS trust store by
default; `SSL_CERT_FILE` / `SSL_CERT_DIR` can supply custom trust on the process
performing the download. Certificate verification remains enabled.

Legacy remote cache acquisition supports both `hf` and `huggingface-cli`. It
retains the older command when both are installed, preserving its argument
format, and pins its isolated `uvx` fallback to the same Hub version as the
manifest adapter.

## Transfer and launch guarantees

Local, push, delegated and per-node pull share preparation. Delegated acquisition
runs on the designated head and need not contact the Hub from the controller.
Each distribution entry has its own revision, including speculative draft models.
Required metadata failures are separate from advisory metadata budgets. Remote
acquisition uses the pinned `huggingface_hub==2.2.0` adapter; offline observation
requires only Bash and ordinary Linux utilities, not Python or uv on targets.

Copies enumerate selected snapshot entries and dereference them into regular
snapshot files. They exclude other revisions, live blob-directory churn, locks,
partial downloads and unrelated cache state. This representation works alongside
classic and shared-blob HF layouts, without copying both snapshots and blobs.
Repair stages one affected file and atomically replaces its snapshot entry; it
never deletes a shared canonical blob. Copied files are not skipped based only
on equal sizes when a checksum repair has identified corruption.

Core holds per-cache/repository `flock` leases across acquisition and provider
calls. Concurrent preparations wait for a busy lease, reporting the holder's
controller, PID and operation every ten seconds, for up to
`model_cache_lease_timeout_s` (cluster `distribution.model.lease_timeout_s`;
default 86400, `0` waits without limit). Because a lease spans the holder's whole
download, a healthy wait can last hours; the ceiling only fails a launch queued
behind a holder that is stuck, and its error names the holder and the setting.
Waiting remains cancellable and continues monitoring leases already acquired;
failure to acquire is reported separately from losing a held lease.

Preparation first locks and checks only its targets, including reuse of existing
blobs. A warm launch, or a push whose head already holds the model, does not take
an unused control-cache lease. If a control source is needed, preparation releases
its target leases, reacquires the full set in order, and repeats validation and
reuse to account for changes during the gap. Leases remain exclusive: transfers
sharing a source still serialize, while an independent warm target can launch.
Metadata writes, repairs, and eviction retain the same locking contract.

An operation nonce recognizes leases reached through shared-cache aliases, both
before locking and when an alias finds the lease busy and reads our own nonce as
its holder. The holder writes the nonce through a separate, fsynced write, since
NFS shows other clients only data the writer has flushed;
complete aliases are omitted from transfer after observation. Core monitors the
holder processes and treats lease loss as terminal: streaming commands are
cancelled and providers receive cancellation through their request. Local Hub
downloads check cancellation between files and before publishing a staged file.
Manifest, usage/claim and revision-ref writes run through the holder, inheriting
its actual lock until each write finishes; a dead holder cannot authorize new
publication. Borrowed transport sessions remain open for their other callers.
External downloaders
and garbage collectors do not participate in these leases. A final observation is
still required, and no receipt promises protection against later external changes.

The reproduced self-copy defect underlying the PR #308 review is also guarded in
the existing local rsync wrapper using directory identity, including symlink
aliases. Vanished-source retries are bounded to one and require a still-complete
manifest source. Diagnostics from the first failure remain logged, the original
time budget remains in force, and target validation is mandatory after success.
`SPARKRUN_NO_RSYNC_RETRY=1` disables retries. The resilience idea follows
[Aisoipheo's PR #308](https://github.com/spark-arena/sparkrun/pull/308); this does
not import that PR's observed-file verifier or repo-wide success marker.

Standard vLLM and SGLang commands consume the prepared snapshot while retaining
the recipe's served model name and immutable revision. Draft revision pins remain
independent. Custom commands should consume `{model}` or the validated revision.
For compatibility with existing v1 and v2 recipes, a literal model argument that
exactly matches the recipe's `model` and omits a revision is allowed with a warning:
the engine may load a different cached revision. A matching served-model alias or
tokenizer argument does not qualify for this exception. Explicit revision conflicts
remain errors. Env `launch.model_path` and
`launch.model_revision` consume the same binding. Hooks additionally receive
`SPARKRUN_MODEL_PATH` (workload path, blank for controller post-commands) and the
resolved `SPARKRUN_MODEL_REVISION`. Compilation/runtime caches remain separate
and are not distributed with model payloads.

For a resolved local model with no known revision, env `{launch.model_revision}` renders as `""`.
Known revisions from validated manifests remain available. Missing model resolution is still an error;
dry runs retain the explicit `<unresolved:launch.model_revision>` placeholder. An empty revision does
not provide a distinct cache namespace for locally modified weights.

Docker access checks run the selected image with its final UID and structured
mounts, without GPU allocation or the image entrypoint. Local checks use the
host execution identity. Unavailable or shadowed mounts fail before eviction.
Custom executors need an equivalent `model_access_command` capability; raw mount
options cannot bypass mapping checks. Delegating runtimes that own preparation
retain that responsibility. ColdSnap native restores still bypass HF preparation
through their asset policy; capture/safetensors preparation uses the normal path.

## Rollout evidence and remaining acceptance

Tiny real-rsync fixtures cover missing files, dangling shared links, aliases,
same-size repair, offline reuse and all four modes. Provider tests reject false
success, changed identities and incomplete outcomes. Workload tests cover pinned
commands, env reuse, UID options and mount shadowing.

Local warm preparation (three samples, two observations per host) measured median
41 ms / 60 ms / 186 ms for 10 / 100 / 1,000 selected files using saved pinned
inventory under the default compatibility policy. It hashes no weights
and batches metadata calls; these are local filesystem measurements, not NFS or
multi-node startup promises. Actual NFS locking/latency and large-model startup
remain deployment acceptance checks before removing the default compatibility fallback.

## Retention report contract and lifecycle

Python retention methods return one `CacheReport` per cache location, with typed
`CacheEntry` entries; mapping access is also supported. `host` is the host the
action ran on; `location` (`CacheLocation`: `kind` network/local/unknown,
`source`, `hosts`, per-host `paths`, `verified`) names every host that shares
that cache. CLI `--json` and daemon results use the
same fields. `reason_code` is the stable machine-readable eligibility outcome;
`reason` is explanatory text. `budget_scope: private_managed_snapshots` makes the
scope of `max_bytes` explicit. `reclaimed_bytes_kind: allocated_bytes_estimate`
distinguishes removed allocation estimates from actual filesystem free space;
`free_bytes_after_kind` identifies a dry-run projection or measured result.
A successful prune operation can return `target_met: false` when protection or
shared blobs prevent reaching the requested target. CLI reports that condition
with a nonzero exit; API clients inspect `target_met` independently of job success.
Model-origin operations accept `endpoint` consistently through Python and daemon
requests; pin/reconcile reject an origin mismatch against the saved receipt.

Confirmed stop, stop-all and stop-after-hooks share claim release and metadata
finalization. Failed or partial teardown preserves recovery metadata. Age-based
metadata pruning never discards outstanding claim references. Cache claims have
no expiry: an unavailable controller is not proof of a stopped workload.

A model revision identifies an immutable origin commit. A plugin generation
identifies one managed plugin installation. Neither is a daemon job ID. Cache
claims protect workload usage on cache hosts; runtime leases pin controller code,
and operation reservations coordinate concurrent mutations. Their storage and
lifetimes remain separate.

Cache `pin` and `reconcile` also accept `--endpoint` (or `HF_ENDPOINT`), matching
the model-origin argument used by inspection and repair. A different origin cannot
mutate another origin's usage receipt merely because repo/revision names match.
