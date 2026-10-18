# Actual Dogwood / pinned Box demo feasibility

Research date: 2026-10-13 UTC. Bounded sidecar: primary-source inspection only. No package installs, Docker pulls/runs, model invocations, cloud provisioning, application-repository edits, or sandbox tests were performed. The executable examples below are proposed, source-checked code, not reported execution results.

## Decision

Use **actual `dogwood-local-engine =1.0.0` inside the official Rust Docker image** for the Linux CPU-only demonstration. This host has a working Docker daemon, so it does not need a host Rust/Cargo installation. Build a small standalone Rust harness, then run the resulting executable inside the same image with networking disabled. This exercises the same published engine/version that Box pins, including durable history and restart recovery. It is **not** a Strands Box OS-sandbox test. [Engine manifest][engine-cargo] [Box engine dependencies][box-policy-cargo]

For **full Box**, use an Apple-silicon Mac with macOS 15+ and build exactly `223e3959d539110eaacfacb6f973c03928dc39b0`. That commit explicitly refuses Linux x86_64. Docker does not remove that architecture gate. The existing release binary is `v0.1.0`, not evidence of an exact build of the requested later commit. [Pinned README][box-readme] [Containment selector][box-facade] [Release API][box-releases]

### Locally observed prerequisites

Read-only commands returned:

```text
uname -sm                         Linux x86_64
getconf GNU_LIBC_VERSION           glibc 2.34
command -v rustc cargo             neither found
command -v node npm python3        available
command -v cc gcc docker           available
docker version                    reachable Linux/amd64 server 25.0.16
docker image ls | engine-name filter
                                  no existing rust/dogwood image found
```

These are local observations, not upstream support claims. The Docker image below still needs to be downloaded by the main agent if its implementation authority permits that; this sidecar did not download it.

## Version and package evidence

| Component | Verified pin / evidence | Practical consequence |
|---|---|---|
| Box source | `223e3959d539110eaacfacb6f973c03928dc39b0`; commit timestamp `2026-10-12T20:07:37Z` | Use source build for this exact revision. [Commit API][box-commit] |
| Box compiler | `rust-toolchain.toml`: Rust `1.98.1` | Keep Mac build and Linux harness compiler aligned. [Toolchain][box-toolchain] |
| Box policy engine | `dogwood-language =1.0.0`, `dogwood-local-engine =1.0.0`, `cedar-policy =4.11.0` | Published registry dependencies, not a mock implementation. [Manifest][box-policy-cargo] |
| Box lockfile | language checksum `6996ef17ca5d06cea0e53bbffb177690e8e4a7f3bbcdb5bf67ad59603f2bf079`; engine checksum `02d82f08653e0090010cfa076bcec582d3aa31e1dc3f0574e4e929c63253878b`; `redb 4.2.0` | Record these and retain the standalone harness's generated lockfile. Top-level pins do not reproduce every Box transitive dependency automatically. [Lockfile][box-lock] |
| Official engine repository | tag `v1.0.0` → public mirror commit `d2cba92a6d33119c1f76e2b01b80802b331c7830` | Official source and API examples are pinned here. [Tags API][engine-tags] |
| Registry source provenance | engine crate `.cargo_vcs_info.json` says `a0f75f92aeb5d8ab5cd5e77aa229a515c3257cf9`; language crate says `996d756de1013b7ae209a14f566a80375a59f2f0` | Do not confuse the published crate's embedded VCS SHA with the public mirror tag SHA. These fields were inspected directly from the `.crate` archives. [Engine archive][engine-archive] [Language archive][language-archive] |
| Official Rust container | `rust:1.98.1-bookworm`; index digest `sha256:93ce27a88655056a51dbdd8f5f2d7ddc071c7b0070fb288a37b5a285fc83971e`; amd64 manifest `sha256:c49256cbe5ea0188bc658a689500d70c41eb51f009a7a7be209caf60a944f3ec` | Verified Docker Hub metadata provides a no-host-Rust compilation path. Run the binary **inside** this image; do not assume its glibc-linked executable will run directly on this glibc-2.34 host. [Docker Hub tag API][rust-image] [Official-image catalog][rust-official] |

### Why not a JS/Python/native-package shortcut?

The official Local Engine repository contains the Rust library, an unpublished Rust `dogwood-server` demo, and a benchmark crate. Its release API returned `[]`; its tree and manifests did not expose an official Python/JS binding or downloadable prebuilt native engine. This is a bounded “not found in these primary distribution surfaces” conclusion, not proof no third party has ever packaged it. The engine's CI builds/tests on `ubuntu-latest`; it has no Box-style aarch64-only containment selector. [Official workspace][engine-workspace] [Server manifest][server-cargo] [Releases API][engine-releases] [CI][engine-ci]

`dogwood-py==0.0.12` **does** have a CPython-3.10-ABI3 Linux x86_64 `manylinux_2_17` wheel. However, its own published description identifies it as unofficial and experimental. Its source distribution's `rust/Cargo.toml` depends on `amzn-dogwood-language` from an unpinned Git URL and contains **no `dogwood-local-engine` dependency**. It wraps the older reference interpreter; it must not be presented as the engine Box uses, or as Box containment. Its GitHub repository links returned 404 during this inspection, but the PyPI metadata and source archive were retrievable. [Versioned PyPI metadata][python-metadata] [Source archive][python-source]

The npm registry search for `dogwood` returned unrelated design systems, a Merkle-tree package, and adjacent governance packages, not an official Local Engine distribution. PyPI's `dogwood-local-engine` JSON endpoint returned 404; the package named `dogwood` is an unrelated PhD project. Neither npm/PyPI search is an exhaustive ecosystem guarantee. [npm query][npm-search] [PyPI engine query][python-engine-query] [Unrelated Python name][python-other]

**Concrete recommendation:** do not install `dogwood-py` for the requested durable-engine experiment. Use the verified Rust image and exact registry crate pins instead.

## Actual Local Engine API and semantics

The relevant public API is:

```rust
DurableTemporalEngine::open(path, snapshot_interval)
engine.install(policy_source, action_schema, event_schema, macro_library)
engine.submit(EventBuilder)
engine.has_policy()
engine.checkpoint()
```

`submit` returns `Submitted { offset, ts, outcome }`; `ts` is assigned by the engine in epoch nanoseconds. `Outcome::Decision(verdict)` applies to `request`; `Outcome::Recorded` applies to history events such as `response`. The event's logged fields (`.field(...)`) and decision context (`.request_context(...)`) are distinct bags; populate both when a value participates in both temporal predicates and Cedar conditions. [Public API walkthrough][engine-guide] [Durable implementation][engine-durable]

Durable submits append/sync before evaluation. Engine verdicts do not execute actions, authenticate event provenance, or prevent arbitrary native syscalls. The wrapping harness must run effects only when allowed and submit genuine outcomes. This is the critical distinction between a policy-engine demo and Box. [Embedding obligations][engine-root] [Engine API walkthrough][engine-guide]

**Recovery pitfall:** reopen the same store without blindly calling `install` again. The implementation documents `install` as `[SetActionSchema; DeleteAll; Add each]`, creating fresh policy instances. Box instead reconciles existing policy content using `batch` to preserve unchanged policies' state. A “restart” that reinstalls every rule can reset precisely the history the test is meant to verify. [Install implementation][engine-durable] [Box reconciliation][box-dogwood]

### Minimal Linux harness: real engine, real dummy-file read, no upload

Create a **new demo directory**, not an application repository. Put this in `Cargo.toml`:

```toml
[package]
name = "dogwood-local-demo"
version = "0.1.0"
edition = "2024"

[dependencies]
dogwood-language = "=1.0.0"
dogwood-local-engine = "=1.0.0"
cedar-policy = "=4.11.0"
redb = "=4.2.0"
```

These are the Box engine/Cedar/redb pins, not a claim that all transitives will match Box's lockfile. The first build generates the demo's `Cargo.lock`; retain it and use `--locked` thereafter. [Box manifest][box-policy-cargo] [Box lockfile][box-lock]

Put the following **new harness code** in `src/main.rs`. It uses official public APIs, but is authored here rather than copied from an upstream example. Prepare `sensitive.txt` with harmless fixture bytes, e.g. `DUMMY_SENSITIVE_MARKER`, before running it.

```rust
use dogwood_language::{Event, EventBuilder, Value};
use dogwood_local_engine::{DurableTemporalEngine, Outcome};
use std::{error::Error, time::Instant};

const SCHEMA: &str = r#"
namespace Demo {
    entity Agent;
    entity Workspace;
    action "Read" appliesTo {
        principal: [Agent], resource: [Workspace],
        context: { input: { path: String } }
    };
    action "Upload" appliesTo {
        principal: [Agent], resource: [Workspace],
        context: { input: { path: String } }
    };
}
"#;

const POLICY: &str = r#"
@id("allow_read")
permit (principal, action == Demo::Action::"Read", resource);
@id("allow_upload")
permit (principal, action == Demo::Action::"Upload", resource);
@id("no_upload_after_sensitive_read")
forbid (principal, action == Demo::Action::"Upload", resource)
when temporal {
    formerly within 1h
    Demo::Action::"Read"::response{ input.path: "sensitive.txt" }
};
"#;

fn event(action: &str, kind: &str, path: &str) -> EventBuilder {
    Event::builder(action, kind)
        .principal("Demo::Agent::\"self\"")
        .resource("Demo::Workspace::\"demo\"")
        .field("input", "path", Value::String(path.to_owned()))
        .request_context("input", "path", Value::String(path.to_owned()))
}

fn request(engine: &mut DurableTemporalEngine, action: &str, path: &str)
    -> Result<bool, Box<dyn Error>>
{
    let result = engine.submit(event(action, "request", path))?;
    match result.outcome {
        Outcome::Decision(verdict) => {
            assert!(verdict.diagnostics().errors().next().is_none(),
                "an evaluator error must not masquerade as a policy denial");
            Ok(verdict.allowed())
        }
        Outcome::Recorded => Err("request unexpectedly returned no verdict".into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: dogwood-local-demo STORE DUMMY_FILE SAMPLES".into());
    }
    let samples: usize = args[3].parse()?;
    if samples == 0 { return Err("SAMPLES must be positive".into()); }
    let mut engine = DurableTemporalEngine::open(&args[1], 1000)?;
    if engine.has_policy() {
        return Err("use a fresh STORE for this scenario, not an existing history".into());
    }
    engine.install(POLICY, SCHEMA, None, None)?;

    assert!(request(&mut engine, "Demo::Action::Upload", "payload")?);
    println!("before_sensitive_read=ALLOW (verdict only; no upload executed)");

    // A request alone is NOT evidence that the read completed.
    assert!(request(&mut engine, "Demo::Action::Read", "sensitive.txt")?);
    assert!(request(&mut engine, "Demo::Action::Upload", "payload")?);
    println!("after_read_request_only=ALLOW");

    // Trusted harness performs a real read of a harmless fixture, then records it.
    let bytes = std::fs::read(&args[2])?;
    println!("dummy_file_bytes={}", bytes.len());
    let recorded = engine.submit(event(
        "Demo::Action::Read", "response", "sensitive.txt"
    ))?;
    assert!(matches!(recorded.outcome, Outcome::Recorded));

    assert!(!request(&mut engine, "Demo::Action::Upload", "payload")?);
    println!("after_completed_sensitive_read=DENY");
    drop(engine);

    // Important: reopen, do NOT reinstall.
    let mut recovered = DurableTemporalEngine::open(&args[1], 1000)?;
    assert!(recovered.has_policy());
    assert!(!request(&mut recovered, "Demo::Action::Upload", "payload")?);
    println!("after_reopen=DENY");

    // Sequential deny-path submit latency: includes durable append/evaluation
    // and any automatic checkpoint, but not compilation/open/install/file read.
    let mut micros = Vec::with_capacity(samples);
    let total = Instant::now();
    for _ in 0..samples {
        let started = Instant::now();
        let result = recovered.submit(event(
            "Demo::Action::Upload", "request", "payload"
        ))?;
        let elapsed = started.elapsed().as_micros();
        match result.outcome {
            Outcome::Decision(verdict) => {
                assert!(verdict.diagnostics().errors().next().is_none());
                assert!(!verdict.allowed());
            }
            Outcome::Recorded => panic!("benchmark request produced no verdict"),
        }
        micros.push(elapsed);
    }
    let seconds = total.elapsed().as_secs_f64();
    micros.sort_unstable();
    let percentile = |p: usize| micros[((micros.len() - 1) * p) / 100];
    println!(
        "samples={} p50_us={} p95_us={} p99_us={} sequential_submits_per_s={:.2}",
        samples, percentile(50), percentile(95), percentile(99),
        samples as f64 / seconds
    );
    Ok(())
}
```

The temporal clause is a `forbid` over a broad permit, keyed on a completed `::response`. Its intended assertion sequence is `ALLOW → ALLOW after request only → DENY after response → DENY after reopen`. That is an expectation until the main agent actually compiles/runs it. This code reads fixture bytes, but does not implement HTTP upload, OS containment, or independently verify a malicious agent's events.

From **only the new demo directory**, the main agent can build/run without host Rust:

```bash
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --env CARGO_HOME=/demo/.cargo \
  --env CARGO_TARGET_DIR=/demo/target \
  --mount "type=bind,src=$PWD,dst=/demo" --workdir /demo \
  rust:1.98.1-bookworm@sha256:93ce27a88655056a51dbdd8f5f2d7ddc071c7b0070fb288a37b5a285fc83971e \
  cargo build --release

docker run --rm --platform linux/amd64 --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/demo" --workdir /demo \
  rust:1.98.1-bookworm@sha256:93ce27a88655056a51dbdd8f5f2d7ddc071c7b0070fb288a37b5a285fc83971e \
  /demo/target/release/dogwood-local-demo \
  /demo/store.redb /demo/sensitive.txt 1000
```

The first command downloads the compiler image and Cargo dependencies if absent; the second runs the actual engine offline. Mount only this demo directory, not the workspace root, home, Docker socket, or credentials. This use of Docker is packaging/build convenience, **not a substitute measurement of Box sandbox enforcement**. [Image metadata][rust-image]

### Official server and temporal example, if a CLI is preferred

The actual official `dogwood-server` exists, but `publish = false`: it must be built from the engine repository. It offers Unix-socket control/data planes, not an HTTP endpoint and not `pip install dogwood-server`. Its CLI is implemented in `dogwood-server/src/main.rs`. [Server manifest][server-cargo] [CLI source][server-main]

After compiling that pinned repository, these are exact CLI shapes:

```bash
# Terminal A; keep this process alive.
./target/release/dogwood-server --dir /absolute/demo-state run

# Terminal B, after the data/control sockets are ready.
./target/release/dogwood-server --dir /absolute/demo-state ping
./target/release/dogwood-server --dir /absolute/demo-state \
  policy install /absolute/policy.dw \
  --action-schema /absolute/schema.cedarschema

./target/release/dogwood-server --dir /absolute/demo-state \
  submit 'Drupe::Action::Login' response \
  --principal 'Drupe::OAuthUser::"alice"' \
  --resource 'Drupe::Gateway::"gw1"' \
  --logged '{"input":{"user":"alice"}}' \
  --context '{"input":{"user":"alice"}}'

./target/release/dogwood-server --dir /absolute/demo-state \
  submit 'Drupe::Action::Read' request \
  --principal 'Drupe::OAuthUser::"alice"' \
  --resource 'Drupe::Gateway::"gw1"' \
  --logged '{"input":{"user":"alice"}}' \
  --context '{"input":{"user":"alice"}}'
```

The matching first-party example policy correlates the current user's request with `formerly within 1h Drupe::Action::"Login"::response{ input.user: context.input.user }`; the complete compatible schema is in the pinned library walkthrough. Responses can be submitted by a trusted harness without a preceding request; that is why this sequence is a policy demonstration, not a claim about authenticated login. A denied `submit` exits nonzero. The same-UID demo is not tamper-resistant; the source requires the monitored agent to have a different, lower-privileged UID for that boundary. [Walkthrough][engine-guide] [CLI/source boundary][server-main]

## Pinned Box platform / preflight behavior

1. **Release support:** pinned README says Apple silicon/macOS 15+. `download.sh` refuses any non-Darwin OS before fetching, then refuses non-arm64/non-aarch64 architectures. It selects only `aarch64-apple-darwin`. GitHub release metadata lists only the macOS `v0.1.0` tarball and checksums. Do not present `BOX_VERSION=v0.1.0` as an exact build of `223e3959…`. [README][box-readme] [Downloader][box-download] [Release API][box-releases]
2. **Source containment:** `linux_mechanism` selects the namespace backend only for `architecture == "aarch64"`, otherwise returns `PlatformUnsupported` with `Linux containment supports ARM64 (aarch64), not x86_64`. For ARM64, `checked_backend` also refuses an unavailable backend after a live user-namespace probe; failure is closed, not uncontained execution. [Selector][box-facade] [Probe][box-linux-probe]
3. **Mac nuance:** documented support and downloader gates are stronger claims than the source facade. The Mac facade is selected by `target_os = "macos"` and Seatbelt's `support_info` simply says available; these are not an independent macOS-version/Apple-silicon qualification check. Use the documented supported host and do a live positive control. [Facade][box-facade] [Seatbelt source][box-seatbelt]
4. **Source/test preflight is not a product `box preflight` command.** `test-integ` creates a fixture and first runs `/bin/echo DET_PREFLIGHT_OK`, refusing to treat later failures as policy denials if this positive control never launched. Assertions additionally require native `DET_ENTERED`, mediated `DET_MEDIATED`, and a journaled Shell decision. On x86_64, the suite documents fixture errors, not successful sandbox denial tests. [Harness README][box-integ-readme] [Harness implementation][box-integ-lib]
5. **Beware a green but vacuous unit/integration-test invocation.** Many `crates/box/tests/box_shell.rs` tests call `namespace_launcher_is_usable()` and `return` when false. That helper immediately returns false on Linux non-aarch64, so `cargo test` on x86_64 may show an apparent passed test whose body never ran. Off Linux the helper returns true. Verify markers/verdicts rather than treating the exit status alone as evidence. [Helper][box-fixture] [Cross-boundary tests][box-shell-tests]
6. **Configuration preflight:** malformed config/policy refuses before the workload; missing paths, symlink/canonical-path disagreements, absent `box_dir` parent, forbidden grants and workspace/state overlap can also stop startup. State directories may remain after refusal. Nonzero startup is not an agent-action denial. [Configuration contract][box-config] [Policy load failures][box-user-policy]

## Runnable full-Box Mac path

### A. Run the existing first-party cross-boundary tests

On an already prepared **Apple-silicon Mac, macOS 15+**, use a dedicated source checkout pinned to the requested commit. Commands below are proposed, not executed here. Rustup/toolchain setup is a separate prerequisite.

```bash
git clone https://github.com/strands-agents/box.git box-pinned-demo-source
cd box-pinned-demo-source
git checkout --detach 223e3959d539110eaacfacb6f973c03928dc39b0
test "$(git rev-parse HEAD)" = 223e3959d539110eaacfacb6f973c03928dc39b0
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
sw_vers -productVersion

cargo build --locked -p strands-box -p strands-box-containment \
  --features test-support

cargo test --locked -p strands-box --features test-support \
  --test box_shell reading_a_secret_through_the_alias_closes_egress \
  -- --exact --nocapture

cargo test --locked -p strands-box --features test-support \
  --test box_shell reading_an_unrelated_file_leaves_egress_open \
  -- --exact --nocapture

cargo test --locked -p strands-box --features test-support \
  --test box_shell a_python_read_closes_egress \
  -- --exact --nocapture
```

Those are real Box runtime tests: a contained workload makes a gateway request, invokes Shell or Monty to read the dummy secret, then makes the same request in the same run. The source asserts `read=ok`, no initial `status=403`, and `status_after=403`; the unrelated-read control asserts egress remains unblocked. No AWS credentials or paid model is needed. [Tests and exact function names][box-shell-tests] [Native egress probe][box-egress-probe]

**Important interpretation:** these upstream tests use `localhost:9`, deliberately without an upstream listener. They establish a **policy-permitted connection decision before the read** versus a gateway `403` after it; they do **not** establish a successful HTTP upload before the read. Report an initial connection/upstream error separately. For delivery-level evidence, a host-side recorder and positive control are required; the repository's `CN-EG-01` fixture demonstrates real upstream counts. [Cross-boundary test comments][box-shell-tests] [Recorder][box-egress-recorder] [Delivery-count fixture][box-egress-granularity]

Do not improvise `curl http://127.0.0.1:<port>` through Strands Shell for that recorder: its own private/loopback-address floor refuses this before the gateway. The existing contained native/MCP proxy-aware probes are the appropriate paths. [Shell curl negative case][box-shell-curl-floor]

### B. Small self-contained full-Box file-effect demo, with native-read control

For a clearer executable demonstration with successful/absent filesystem effects, build the pinned product binaries:

```bash
cargo build --locked --release -p strands-box -p strands-box-containment
```

Keep `strands-box`, its alias image, and containment trampoline together in `target/release`; helper lookup matters. [Build documentation][box-readme] [Helper requirement][box-integ-readme]

In a **new** canonical-path demo root named `box-primary-demo` directly under the operator's home, with sibling directories `control/`, `workspace/`, and `box-state/`, create a harmless `workspace/sensitive.txt` containing `DUMMY_SENSITIVE_MARKER`. The sensitive path will therefore be reported to policy as `~/box-primary-demo/workspace/sensitive.txt`. Put this in `control/box.toml`, replacing `<ROOT>` with its real absolute path:

```toml
name = "sensitive-read-demo"
box_dir = "<ROOT>/box-state/one"
policy = "policy.dw"

[agent]
workspace = "<ROOT>/workspace"
command = ["/bin/bash", "-c", '''
printf 'NATIVE_ENTERED\n'
if IFS= read -r native_line < sensitive.txt; then
  printf 'DIRECT_READ_BYPASS\n'
  exit 10
else
  printf 'DIRECT_READ_BLOCKED\n'
fi
zsh -lc 'echo MEDIATED_ENTERED; printf before > before.txt; rc=$?; echo BEFORE_WRITE_RC=$rc; test "$rc" -eq 0' || exit 11
zsh -lc 'cat sensitive.txt; rc=$?; echo READ_RC=$rc; test "$rc" -eq 0' || exit 12
zsh -lc 'printf after > after.txt; rc=$?; echo AFTER_WRITE_RC=$rc; test "$rc" -ne 0' || exit 13
printf 'DEMO_DONE\n'
''']
env = { PATH = "/usr/bin:/bin" }

# Deliberately no [agent.filesystem] read/write grant to workspace.
```

Put this in `control/policy.dw`:

```text
@id("demo_shell")
permit (principal, action == Box::Action::"shell:exec", resource);
@id("demo_read")
permit (principal, action == Box::Action::"fs:read", resource);
@id("demo_write")
permit (principal, action == Box::Action::"fs:write", resource);

@id("no_writes_after_sensitive")
@description("A completed sensitive read closes content writes for one hour.")
forbid (principal, action == Box::Action::"fs:write", resource)
when { context.input.operation == Box::FsWriteOperation::"write_content" }
when temporal {
    formerly within 1h
    Box::Action::"fs:read"::response{
        input.path: "~/box-primary-demo/workspace/sensitive.txt",
        input.operation: Box::FsReadOperation::"read_content"
    }
};
```

The first three permits are deliberately broad demo permissions for broker-managed operations, subject to Box's mandatory reach floor; they are **not a production sensitive-data policy**. Native workspace reads are still ungranted. If the demo root has a different name or location, replace the literal sensitive path with the exact policy-facing spelling, not a raw `/Users/...` spelling under the operator home. This combines documented native-vs-mediated paths with the response-keyed temporal condition used by upstream tests. [Shell invocation and operation fields][box-user-shell] [Containment/config distinction][box-user-policy] [Response-keyed tests][box-policy-shell]

Run from the host:

```bash
/absolute/box-pinned-demo-source/target/release/strands-box \
  run --config /absolute/demo-root/control/box.toml
```

Acceptance criteria, to be checked by the main runner:

Syntax-only local check: the proposed TOML parsed with Python's standard `tomllib`, and its extracted native Bash command passed `bash -n`. This does not validate Dogwood compilation or execute the Mac sandbox.

- `NATIVE_ENTERED`, `DIRECT_READ_BLOCKED`, `MEDIATED_ENTERED`, and `DEMO_DONE` appear; `DIRECT_READ_BYPASS` does not.
- `BEFORE_WRITE_RC=0`, `READ_RC=0`, and `DUMMY_SENSITIVE_MARKER` appear.
- The final write returns nonzero and its error names `no_writes_after_sensitive`.
- Host-side `workspace/before.txt` exists and contains `before`; `workspace/after.txt` is absent.
- `<ROOT>/box-state/one/private/telemetry/records.jsonl` contains `fs:read permit` and later `fs:write deny` governed by `no_writes_after_sensitive`; the native failed read should have no `fs:read` decision.
- Repeat the write with the same Box state before the one-hour window expires: it should remain denied. Use a **fresh state directory** for a fresh allow-before-read run; do not delete a live box.

This proposal is a **full-Box OS containment + broker + actual engine** test, but the effect being denied is a file write, not an HTTP upload. Pair it with the first-party egress tests when explaining cross-lane governance. Since it has not been executed here, startup or policy compilation failures must be reported as setup errors, never relabeled as successful security denials.

## Performance: what can honestly be measured

The standalone harness reports **sequential durable `submit` latency for one deny-path temporal policy**, not full Box end-to-end latency and not GPU/model inference. Its timing includes log durability and automatic snapshots, excludes compilation/install/open/read, and has no concurrent load. Preserve CPU/kernel/filesystem/storage, image digest, Cargo lockfile, sample count, verdict counts, and snapshot interval beside the results.

For policy-count/concurrency experiments, the official repository includes:

```bash
cargo run --release -p dogwood-performance --bin load -- \
  --policies 10,100,1000 --rates 1,10,100,500 --seconds 5 --clients 16
```

This requires building the larger upstream benchmark workspace; it is not needed for the minimal demonstration. The published harness explicitly separates the in-memory reference from the durable server and warns that `fsync` sets a storage floor. Its published machine-specific results are **not measurements from this host**. More CPUs do not remove one store's serialized durable append point. [Benchmark methodology][engine-performance] [Durable submit implementation][engine-durable]

## Source index

[engine-root]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/README.md
[engine-guide]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/dogwood-local-engine/README.md
[engine-cargo]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/dogwood-local-engine/Cargo.toml
[engine-workspace]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/Cargo.toml
[engine-durable]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/dogwood-local-engine/src/durable.rs
[engine-performance]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/dogwood-performance/README.md
[engine-ci]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/.github/workflows/ci.yml
[engine-tags]: https://api.github.com/repos/dogwood-policy/dogwood-local-engine/tags
[engine-releases]: https://api.github.com/repos/dogwood-policy/dogwood-local-engine/releases
[engine-archive]: https://static.crates.io/crates/dogwood-local-engine/dogwood-local-engine-1.0.0.crate
[language-archive]: https://static.crates.io/crates/dogwood-language/dogwood-language-1.0.0.crate
[server-cargo]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/dogwood-server/Cargo.toml
[server-main]: https://github.com/dogwood-policy/dogwood-local-engine/blob/d2cba92a6d33119c1f76e2b01b80802b331c7830/dogwood-server/src/main.rs
[rust-image]: https://hub.docker.com/v2/repositories/library/rust/tags/1.98.1-bookworm
[rust-official]: https://github.com/docker-library/official-images/blob/master/library/rust
[python-metadata]: https://pypi.org/pypi/dogwood-py/0.0.12/json
[python-source]: https://files.pythonhosted.org/packages/90/3c/04f580639bea989dc62abe6bcbb6b4ccecd6501bc1afc37626476cb2020a/dogwood_py-0.0.12.tar.gz
[npm-search]: https://registry.npmjs.org/-/v1/search?text=dogwood&size=50
[python-engine-query]: https://pypi.org/pypi/dogwood-local-engine/json
[python-other]: https://pypi.org/pypi/dogwood/json
[box-commit]: https://api.github.com/repos/strands-agents/box/commits/223e3959d539110eaacfacb6f973c03928dc39b0
[box-readme]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/README.md
[box-policy-cargo]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/policy/Cargo.toml
[box-lock]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/Cargo.lock
[box-toolchain]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/rust-toolchain.toml
[box-facade]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/containment/src/facade.rs
[box-linux-probe]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/containment/src/backend/linux/namespace/probe.rs
[box-seatbelt]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/containment/src/backend/macos/seatbelt.rs
[box-download]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/download.sh
[box-releases]: https://api.github.com/repos/strands-agents/box/releases
[box-dogwood]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/policy/src/dogwood.rs
[box-config]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/user/config.md
[box-user-policy]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/user/policy.md
[box-user-shell]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/user/shell.md
[box-policy-shell]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/policy/tests/temporal_shell_e2e.rs
[box-integ-readme]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/test-integ/README.md
[box-integ-lib]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/test-integ/src/lib.rs
[box-fixture]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/box/tests/support/fixture.rs
[box-shell-tests]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/box/tests/box_shell.rs
[box-egress-probe]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/box/tests/support/egress_probe.rs
[box-egress-recorder]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/test-integ/src/egress.rs
[box-egress-granularity]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/test-integ/tests/containment/cn_egress_granularity.rs
[box-shell-curl-floor]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/test-integ/tests/shell/sh_neg_curl_403.rs
