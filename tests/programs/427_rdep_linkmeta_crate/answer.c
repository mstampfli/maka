/* A native symbol provided only through the build script's link flags. If
   makac fails to forward -L/-l from the sidecar's build scripts, the final
   link cannot resolve this and the test fails. */
int link_answer(void) { return 42; }
