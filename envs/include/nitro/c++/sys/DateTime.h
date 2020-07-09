/* =========================================================================
 * This file is part of sys-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2012, General Dynamics - Advanced Information Systems
 *
 * sys-c++ is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this program; If not, 
 * see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef __SYS_DATE_TIME_H__
#define __SYS_DATE_TIME_H__

#include <string>
#include <time.h>

#include "Export.h"

namespace sys
{

/*!
 *  Representation of a date/time structure.
 */
class DLL_PUBLIC_CLASS DateTime
{
protected:
    int mYear;
    int mMonth;
    int mDayOfMonth;
    int mDayOfWeek;
    int mDayOfYear;
    int mHour;
    int mMinute;
    double mSecond;
    double mTimeInMillis;

    // Turn a tm struct into a double
    double toMillis(tm t) const;

    /*! 
     * Set the time to right now.  
     * Uses time() or if HAVE_SYS_TIME_H is defined, 
     * gettimeofday() for usec precision.
     */
    void setNow();

    //! @brief Set members from the millis value.
    void fromMillis();

    //! @brief Set members from the tm struct value.
    virtual void fromMillis(const tm& t);

    //! @brief Set the millis value from the members
    virtual void toMillis() = 0;

    //! @brief Provides the time as a 'tm'
    void getTime(tm& t) const;

    //! @brief Given seconds since the epoch, provides the time
    virtual void getTime(time_t numSecondsSinceEpoch, tm& t) const = 0;

public:
    DateTime();
    virtual ~DateTime();

    //! Return month {1,12}
    int getMonth() const { return mMonth; }
    //! Return day of month {1,31}
    int getDayOfMonth() const { return mDayOfMonth; }
    //! Return day of week {1,7}
    int getDayOfWeek() const { return mDayOfWeek; }
    //! Return day of year {1,366}
    int getDayOfYear() const { return mDayOfYear; }
    //! Return hour {0,23}
    int getHour() const { return mHour; }
    //! Return minute {0,59}
    int getMinute() const { return mMinute; }
    //! Return second {0,59}
    double getSecond() const { return mSecond; }
    //! Return millis since 1 Jan 1970
    double getTimeInMillis() const { return mTimeInMillis; }
    //! Return the current year
    int getYear() const { return mYear; }

    // ! Given the {1,12} month return the alphabetic equivalent
    static std::string monthToString(int month);
    // ! Given the {1,7} day of the week return the alphabetic equivalent
    static std::string dayOfWeekToString(int dayOfWeek);

    // ! Given the {1,12} month return the abbreviated alphabetic equivalent
    static std::string monthToStringAbbr(int month);
    // ! Given the {1,7} day, return the abbreviated alphabetic equivalent
    static std::string dayOfWeekToStringAbbr(int dayOfWeek);

    // ! Given the alphabetic or abbreviated version return {1,12} equivalent 
    // Acceptable input "August" or "Aug" would return 8
    static int monthToValue(const std::string& month);
    // ! Given the alphabetic or abbreviated version return {1,7} equivalent 
    // Acceptable input "Wednesday" or "Wed" would return 4
    static int dayOfWeekToValue(const std::string& dayOfWeek);

    // Setters
    void setMonth(int month);
    void setDayOfMonth(int dayOfMonth);
    void setHour(int hour);
    void setMinute(int minute);
    void setSecond(double second);
    void setTimeInMillis(double time);
    void setYear(int year);

    /*!
     *  format the DateTime string
     *  y = year (YYYY)
     *  M = month (MM)
     *  d = day (DD)
     *  H = hour (hh)
     *  m = minute (mm)
     *  s = second (ss)
     */
    std::string format(const std::string& formatStr) const;

    /**
     * @name Logical Operators.
     * @brief Logical comparison operators.
     *
     * @param rhs The object to compare against.
     *
     * @return true if comparison holds, false otherwise.
     */
    //@{
    bool operator<(const DateTime& rhs) const
    {
        return (mTimeInMillis < rhs.mTimeInMillis);
    }

    bool operator<=(const DateTime& rhs) const
    {
        return (mTimeInMillis <= rhs.mTimeInMillis);
    }

    bool operator>(const DateTime& rhs) const
    {
        return (mTimeInMillis > rhs.mTimeInMillis);
    }

    bool operator>=(const DateTime& rhs) const
    {
        return (mTimeInMillis >= rhs.mTimeInMillis);
    }

    bool operator==(const DateTime& rhs) const
    {
        return (mTimeInMillis == rhs.mTimeInMillis);
    }

    bool operator!=(const DateTime& rhs) const
    {
        return !operator==(rhs);
    }
    //@}
};

}

#endif//__SYS_DATE_TIME_H__
