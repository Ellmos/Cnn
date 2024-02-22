#pragma once

#include <cstdio>

enum LogLevel
{
    NO_LOG,
    INFO,
    TRACE,
    WARN,
    ERROR
};

static const LogLevel level = NO_LOG;

enum TextColor
{
    TEXT_COLOR_BLACK,
    TEXT_COLOR_RED,
    TEXT_COLOR_GREEN,
    TEXT_COLOR_YELLOW,
    TEXT_COLOR_BLUE,
    TEXT_COLOR_MAGENTA,
    TEXT_COLOR_CYAN,
    TEXT_COLOR_WHITE,
    TEXT_COLOR_BRIGHT_BLACK,
    TEXT_COLOR_BRIGHT_RED,
    TEXT_COLOR_BRIGHT_GREEN,
    TEXT_COLOR_BRIGHT_YELLOW,
    TEXT_COLOR_BRIGHT_BLUE,
    TEXT_COLOR_BRIGHT_MAGENTA,
    TEXT_COLOR_BRIGHT_CYAN,
    TEXT_COLOR_BRIGHT_WHITE,
};

// No array designators so don't change order in this or it will go kaput :(
static const char *TextColorTable[] = {
    "\x1b[30m", // TEXT_COLOR_BLACK
    "\x1b[31m", // TEXT_COLOR_RED
    "\x1b[32m", // TEXT_COLOR_GREEN
    "\x1b[33m", // TEXT_COLOR_YELLOW
    "\x1b[34m", // TEXT_COLOR_BLUE
    "\x1b[35m", // TEXT_COLOR_MAGENTA
    "\x1b[36m", // TEXT_COLOR_CYAN
    "\x1b[37m", // TEXT_COLOR_WHITE
    "\x1b[90m", // TEXT_COLOR_BRIGHT_BLACK
    "\x1b[91m", // TEXT_COLOR_BRIGHT_RED
    "\x1b[92m", // TEXT_COLOR_BRIGHT_GREEN
    "\x1b[93m", // TEXT_COLOR_BRIGHT_YELLOW
    "\x1b[94m", // TEXT_COLOR_BRIGHT_BLUE
    "\x1b[95m", // TEXT_COLOR_BRIGHT_MAGENTA
    "\x1b[96m", // TEXT_COLOR_BRIGHT_CYAN
    "\x1b[97m", // TEXT_COLOR_BRIGHT_WHITE
};

template <typename... Args>
void _log(const char *prefix, TextColor textColor, const char *msg,
          Args... args)
{
    char formatBuffer[8192] = {};
    sprintf(formatBuffer, "%s %s %s \033[0m", TextColorTable[textColor], prefix,
            msg);

    char textBuffer[8912] = {};
    sprintf(textBuffer, formatBuffer, args...);

    puts(textBuffer);
}

#define LOG_INFO(...)                                                          \
    if (level == INFO)                                                         \
    {                                                                          \
        _log("INFO: ", TEXT_COLOR_WHITE, __VA_ARGS__);                         \
    }
#define LOG_TRACE(...)                                                         \
    if (level >= TRACE)                                                        \
    {                                                                          \
        _log("TRACE: ", TEXT_COLOR_CYAN, __VA_ARGS__);                         \
    }
#define LOG_WARN(...)                                                          \
    if (level >= WARN)                                                         \
    {                                                                          \
        _log("WARN: ", TEXT_COLOR_YELLOW, __VA_ARGS__);                        \
    }
#define LOG_ERROR(...)                                                         \
    if (level >= ERROR)                                                        \
    {                                                                          \
        _log("ERROR: ", TEXT_COLOR_RED, __VA_ARGS__);                          \
    }
