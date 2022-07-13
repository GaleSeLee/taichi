#define COLOR_DEBUG "\033[36m" /* Cyan */

#define tick                                                                \
    do {                                                                    \
        std::cerr << "[" COLOR_DEBUG "TICK" COLOR_RESET " " << __FILE__ \
                  << ":" << __FUNCTION__ << ":" << __LINE__ << "] "     \
                  << std::endl;                                         \
    } while (0)