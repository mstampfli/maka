use std::process::Command;

// Compile answer.c into a native static lib in OUT_DIR, then tell cargo to
// link it. These two `cargo:` lines are exactly the metadata makac must
// harvest from the sidecar build and forward to the final cc link line.
fn main() {
    println!("cargo:rerun-if-changed=answer.c");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".into());
    let obj = format!("{}/answer.o", out);
    let lib = format!("{}/libanswer.a", out);

    let ok = Command::new(&cc)
        .args(["-c", "answer.c", "-o", &obj])
        .status()
        .expect("failed to spawn C compiler");
    assert!(ok.success(), "compiling answer.c failed");

    let ok = Command::new("ar")
        .args(["rcs", &lib, &obj])
        .status()
        .expect("failed to spawn ar");
    assert!(ok.success(), "archiving libanswer.a failed");

    println!("cargo:rustc-link-search=native={}", out);
    // `-bundle`: do NOT archive answer.o into the sidecar staticlib, so the
    // symbol stays undefined there and the final cc link genuinely depends on
    // makac forwarding this flag (mirrors a real import lib like WebView2Loader).
    println!("cargo:rustc-link-lib=static:-bundle=answer");
}
