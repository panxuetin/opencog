/*
 * opencog/util/Logger.cc
 *
 * Copyright (C) 2002-2007 Novamente LLC
 * Copyright (C) 2008 by OpenCog Foundation
 * Copyright (C) 2009, 2011 Linas Vepstas
 * Copyright (C) 2010 OpenCog Foundation
 * All Rights Reserved
 *
 * Written by Andre Senna <senna@vettalabs.com>
 *            Gustavo Gama <gama@vettalabs.com>
 *            Joel Pitt <joel@opencog.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "Logger.h"
#include "Config.h"

#ifndef WIN32
#include <cxxabi.h>
#include <execinfo.h>
#endif

#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <sstream>

#ifdef WIN32_NOT_UNIX
#include <winsock2.h>
#undef ERROR
#undef DEBUG
#else
#include <sys/time.h>
#endif

#include <boost/algorithm/string.hpp>

#include <opencog/util/platform.h>

using namespace opencog;

// messages greater than this will be truncated
#define MAX_PRINTF_STYLE_MESSAGE_SIZE (1<<15)
const char* levelStrings[] = {"NONE", "ERROR", "WARN", "INFO", "DEBUG", "FINE"};

#ifndef WIN32 /// @todo backtrace and backtrace_symbols is UNIX, we
              /// may need a WIN32 version
static void prt_backtrace(std::ostringstream& oss)
{
#define BT_BUFSZ 50
	void *bt_buf[BT_BUFSZ];

	int stack_depth = backtrace(bt_buf, BT_BUFSZ);
	char **syms = backtrace_symbols(bt_buf, stack_depth);

	// Start printing at a bit into the stack, so as to avoid recording
	// the logger functions in the stack trace.
	oss << "\tStack Trace:\n";
	for (int i=2; i < stack_depth; i++)
	{
		// Most things we'll print are mangled C++ names,
		// So demangle them, get them to pretty-print.
		char * begin = strchr(syms[i], '(');
		char * end = strchr(syms[i], '+');
		if (!(begin && end) || end <= begin)
		{
			// Failed to pull apart the symbol names
            oss << "\t" << i << ": " << syms[i] << "\n";
		}
		else
		{
			*begin = 0x0;
			oss << "\t" << i << ": " << syms[i] << " ";
			*begin = '(';
			size_t sz = 250;
			int status;
			char *fname = (char *) malloc(sz);
			*end = 0x0;
			char *rv = abi::__cxa_demangle(begin+1, fname, &sz, &status);
			*end = '+';
			if (rv) fname = rv; // might have re-alloced
			oss << "(" << fname << " " << end << std::endl;
			free(fname);
		}
	}
	oss << std::endl;
	free(syms);
}
#endif

Logger::~Logger()
{
#ifdef ASYNC_LOGGING
    // Wait for queue to empty
    flush();
    stopWriteLoop();
#endif

    if (f != NULL) fclose(f);
}

#ifdef ASYNC_LOGGING
void Logger::startWriteLoop()
{
    pthread_mutex_lock(&lock);
    if (!writingLoopActive)
    {
        writingLoopActive = true;
        m_Thread = boost::thread(&Logger::writingLoop, this);
    }
    pthread_mutex_unlock(&lock);
}

void Logger::stopWriteLoop()
{
    pthread_mutex_lock(&lock);
    pendingMessagesToWrite.cancel();
    // rejoin thread
    m_Thread.join();
    writingLoopActive = false;
    pthread_mutex_unlock(&lock);
}

void Logger::writingLoop()
{
    try
    {
        while (true)
        {
            // Must not pop until *after* the message has been written,
            // as otherwise, the flush() call will race with the write,
            // causing flush to report an empty queue, even though the
            // message has not actually been written yet.
            std::string* msg;
            pendingMessagesToWrite.wait_and_get(msg);
            writeMsg(*msg);
            pendingMessagesToWrite.pop();
            delete msg;
        }
    }
    catch (concurrent_queue< std::string* >::Canceled &e)
    {
        return;
    }
}

void Logger::flush()
{
    while (!pendingMessagesToWrite.empty())
    {
        pthread_yield();
        usleep(100);
    }
}
#endif

void Logger::writeMsg(std::string &msg)
{
    pthread_mutex_lock(&lock);
    // delay opening the file until the first logging statement is issued;
    // this allows us to set the main logger's filename without creating
    // a useless log file with the default filename
    if (f == NULL)
    {
        if ((f = fopen(fileName.c_str(), "a")) == NULL)
        {
            fprintf(stderr, "[ERROR] Unable to open log file \"%s\"\n",
                    fileName.c_str());
            disable();
            return;
        }
        else
            enable();
    }

    // write to file
    fprintf(f, "%s", msg.c_str());
    fflush(f);
    pthread_mutex_unlock(&lock);

    // write to stdout
    if (printToStdout)
    {
        std::cout << msg;
        std::cout.flush();
    }
}

Logger::Logger(const std::string &fname, Logger::Level level, bool tsEnabled)
    : error(*this), warn(*this), info(*this), debug(*this), fine(*this)
{
    this->fileName.assign(fname);
    this->currentLevel = level;
    this->backTraceLevel = getLevelFromString(opencog::config()["BACK_TRACE_LOG_LEVEL"]);
    this->timestampEnabled = tsEnabled;
    this->printToStdout = false;

    this->logEnabled = true;
    this->f = NULL;

    pthread_mutex_init(&lock, NULL);
#ifdef ASYNC_LOGGING
    this->writingLoopActive = false;
    startWriteLoop();
#endif // ASYNC_LOGGING
}

Logger::Logger(const Logger& log)
    : error(*this), warn(*this), info(*this), debug(*this), fine(*this)
{
    pthread_mutex_init(&lock, NULL);
    set(log);
}

Logger& Logger::operator=(const Logger& log)
{
#ifdef ASYNC_LOGGING
    this->stopWriteLoop();
    pendingMessagesToWrite.cancel_reset();
#endif // ASYNC_LOGGING
    this->set(log);
    return *this;
}

void Logger::set(const Logger& log)
{
    this->fileName.assign(log.fileName);
    this->currentLevel = log.currentLevel;
    this->backTraceLevel = log.backTraceLevel;
    this->timestampEnabled = log.timestampEnabled;
    this->printToStdout = log.printToStdout;

    this->logEnabled = log.logEnabled;
    this->f = log.f;

#ifdef ASYNC_LOGGING
    startWriteLoop();
#endif // ASYNC_LOGGING
}

// ***********************************************/
// API

void Logger::setLevel(Logger::Level newLevel)
{
    currentLevel = newLevel;
}

Logger::Level Logger::getLevel() const
{
    return currentLevel;
}

void Logger::setBackTraceLevel(Logger::Level newLevel)
{
    backTraceLevel = newLevel;
}

Logger::Level Logger::getBackTraceLevel() const
{
    return backTraceLevel;
}

void Logger::setFilename(const std::string& s)
{
    fileName.assign(s);

    pthread_mutex_lock(&lock);
    if (f != NULL) fclose(f);
    f = NULL;
    pthread_mutex_unlock(&lock);

    enable();
}

const std::string& Logger::getFilename()
{
    return fileName;
}

void Logger::setTimestampFlag(bool flag)
{
    timestampEnabled = flag;
}

void Logger::setPrintToStdoutFlag(bool flag)
{
    printToStdout = flag;
}

void Logger::setPrintErrorLevelStdout() {
    setPrintToStdoutFlag(true);
    setLevel(Logger::ERROR);
}

void Logger::enable()
{
    logEnabled = true;
}

void Logger::disable()
{
    logEnabled = false;
}

void Logger::log(Logger::Level level, const std::string &txt)
{
#ifdef ASYNC_LOGGING
    static const unsigned int max_queue_size_allowed = 1024;
#endif
    static char timestamp[64];
    static char timestampStr[256];

    // Don't log if not enabled, or level is too low.
    if (!logEnabled) return;
    if (level > currentLevel) return;

    std::ostringstream oss;
    if (timestampEnabled)
    {
        struct timeval stv;
        struct tm stm;

        ::gettimeofday(&stv, NULL);
        time_t t = stv.tv_sec;
        gmtime_r(&t, &stm);
        strftime(timestamp, sizeof(timestamp), "%F %T", &stm);
        snprintf(timestampStr, sizeof(timestampStr),
                "[%s:%03ld] ",timestamp, stv.tv_usec / 1000);
        oss << timestampStr;
    }

    oss << "[" << getLevelString(level) << "] " << txt << std::endl;

    if (level <= backTraceLevel)
    {
#ifndef WIN32
        prt_backtrace(oss);
#endif
    }
#ifdef ASYNC_LOGGING
    pendingMessagesToWrite.push(new std::string(oss.str()));

    // If the queue gets too full, block until it's flushed to file or
    // stdout
    if (pendingMessagesToWrite.approx_size() > max_queue_size_allowed) {
        flush();
    }
#else
    std::string temp(oss.str());
    writeMsg(temp);
#endif
}

void Logger::logva(Logger::Level level, const char *fmt, va_list args)
{
    if (level <= currentLevel) {
        char buffer[MAX_PRINTF_STYLE_MESSAGE_SIZE];
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        std::string msg = buffer;
        log(level, msg);
    }
}

void Logger::log(Logger::Level level, const char *fmt, ...)
{
    va_list args; va_start(args, fmt); logva(level, fmt, args); va_end(args);
}
void Logger::Error::operator()(const char *fmt, ...)
{
    va_list args; va_start(args, fmt); logger.logva(ERROR, fmt, args); va_end(args);
}
void Logger::Warn::operator()(const char *fmt, ...)
{
    va_list args; va_start(args, fmt); logger.logva(WARN,  fmt, args); va_end(args);
}
void Logger::Info::operator()(const char *fmt, ...)
{
    va_list args; va_start(args, fmt); logger.logva(INFO,  fmt, args); va_end(args);
}
void Logger::Debug::operator()(const char *fmt, ...)
{
    va_list args; va_start(args, fmt); logger.logva(DEBUG, fmt, args); va_end(args);
}
void Logger::Fine::operator()(const char *fmt, ...)
{
    va_list args; va_start(args, fmt); logger.logva(FINE,  fmt, args); va_end(args);
}

bool Logger::isEnabled(Level level) const
{
    return level <= currentLevel;
}

bool Logger::isErrorEnabled() const
{
    return ERROR <= currentLevel;
}

bool Logger::isWarnEnabled() const
{
    return WARN <= currentLevel;
}

bool Logger::isInfoEnabled() const
{
    return INFO <= currentLevel;
}

bool Logger::isDebugEnabled() const
{
    return DEBUG <= currentLevel;
}

bool Logger::isFineEnabled() const
{
    return FINE <= currentLevel;
}


const char* Logger::getLevelString(const Logger::Level level)
{
    if (level == BAD_LEVEL)
        return "Bad level";
    else
        return levelStrings[level];
}

const Logger::Level Logger::getLevelFromString(const std::string& levelStr)
{
    unsigned int nLevels = sizeof(levelStrings) / sizeof(levelStrings[0]);
    for (unsigned int i = 0; i < nLevels; ++i) {
        if (boost::iequals(levelStrings[i], levelStr))
            return (Logger::Level) i;
    }
    return BAD_LEVEL;
}

// create and return the single instance
Logger& opencog::logger()
{
    static Logger instance;
    return instance;
}
