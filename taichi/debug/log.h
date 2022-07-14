#define COLOR_DEBUG "\033[36m" /* Cyan */
#define COLOR_RESET "\033[0m"
#define tick                                                                \
    do {                                                                    \
        std::cerr << "[" COLOR_DEBUG "TICK" COLOR_RESET " " << __FILE__ \
                  << ":" << __FUNCTION__ << ":" << __LINE__ << "] "     \
                  << std::endl;                                         \
    } while (0)

#define tickv(x)                                                              \
    do {                                                                      \
        std::cerr << "[" COLOR_DEBUG "DEBUG" COLOR_RESET " " << __FILE__  \
                  << ":" << __FUNCTION__ << ":" << __LINE__ << "] " << #x \
                  << " = " << (x) << std::endl;                           \
    } while (0)