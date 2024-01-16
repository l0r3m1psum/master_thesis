#include <os/activity.h>
#include <os/log.h>
#include <os/trace.h>

#include <execinfo.h>

#include <printf.h>


void f(void) {
	void* callstack[128];
	int frames = backtrace(callstack, 128);
	int stderr = 2;
	backtrace_symbols_fd(callstack, frames, stderr);
#if 0
	char** strs = backtrace_symbols(callstack, frames);
	for (int i = 0; i < frames; ++i) {
		printf("%s\n", strs[i]);
	}
	free(strs);
#endif
}

// /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/printf.h
// https://www.gnu.org/software/libc/manual/html_node/Printf-Extension-Example.html
int render(FILE *stream, const struct printf_info *info, const void *const *args) {
	return 0;
}
int arginfo(const struct printf_info *__info, size_t __n, int *__argtypes) {
	return 0;
}

int main(void) {
	// mostly outdated informations: https://www.objc.io/issues/19-debugging/activity-tracing/
	// https://blog.kandji.io/mac-logging-and-the-log-command-a-guide-for-apple-admins
	// https://www.jviotti.com/2022/02/21/emitting-signposts-to-instruments-on-macos-using-cpp.html
	// https://mackuba.eu/notes/wwdc16/unified-logging-and-activity-tracing/
	// https://mackuba.eu/notes/wwdc14/using-activity-tracing/
	// https://devstreaming-cdn.apple.com/videos/wwdc/2016/721wh2etddp4ghxhpcg/721/721_unified_logging_and_activity_tracing.pdf
	os_log(OS_LOG_DEFAULT, "Standard log message.");
	os_activity_t activity = os_activity_create(
		"activity deacription", OS_ACTIVITY_NONE, OS_ACTIVITY_FLAG_DEFAULT);
	{
		os_activity_scope(activity);
		os_log(OS_LOG_DEFAULT, "Logging in an activity 1.");
		// It does not shows up in the log.
		os_activity_label_useraction("event description");
		os_log(OS_LOG_DEFAULT, "Logging in an activity 2.");
		os_log_info(OS_LOG_DEFAULT, "Something happened at level %d", 99);
		os_log(OS_LOG_DEFAULT, "Logging in an activity 3.");
		// For some reason the activity is not displayed in lldb.
		// __builtin_debugtrap();
		os_log(OS_LOG_DEFAULT, "Logging in an activity 4.");
	}
	f();

	printf_domain_t domain = new_printf_domain();

	int res = register_printf_domain_function(domain, 'W', render, arginfo, NULL);

	xprintf(domain, NULL, "Hello\n");

	if (res == -1) {
		return 1;
	}
	free_printf_domain(domain);

	return 0;
}

// Stuff for tests
// https://www.iosdev.recipes/os-signpost/
// https://stackoverflow.com/questions/41451356/is-it-possible-to-use-xctest-unit-tests-without-xcode

// Stuff for malloc
// https://www.cocoawithlove.com/2010/05/look-at-how-malloc-works-on-mac.html
// https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/Articles/MemoryAlloc.html
// https://flylib.com/books/en/3.126.1.98/1/
// https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/Articles/MallocDebug.html
// https://www.synacktiv.com/ressources/Sthack_2018_Heapple_Pie.pdf
// https://chromium.googlesource.com/chromium/src/+/9550b71347ce9acbe2875fb75ea46ec096230506/base/allocator/allocator_interception_mac.mm
// http://eatmyrandom.blogspot.com/2010/03/mallocfree-interception-on-mac-os-x.html

// Not entirely related
// https://eclecticlight.co/2020/08/12/how-a-kernel-zone-memory-leak-can-panic-macos/
// https://newosxbook.com/home.html
