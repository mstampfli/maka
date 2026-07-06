extern "C" {
    fn link_answer() -> i32;
}

// Resolves only if the build script's link flags reach the final link.
pub fn answer() -> i32 {
    unsafe { link_answer() }
}
