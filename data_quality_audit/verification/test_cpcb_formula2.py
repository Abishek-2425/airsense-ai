import pandas as pd
import numpy as np

# Official CPCB Sub-index calculation according to CPCB NAQI report (October 2014)
# Formula:
# Ip = [ (I_hi - I_lo) / (B_hi - B_lo) ] * (Cp - B_lo) + I_lo
#
# Breakpoints:
# PM2.5 (24-hr avg):
#   0-30: 0-50
#   31-60: 51-100
#   61-90: 101-200
#   91-120: 201-300
#   121-250: 301-400
#   251-380 (or 250+): wait, what did CPCB specify?
# Let's inspect the exact formula used in python AQI calculations for India air quality.

def get_sub_index(Cp, breakpoints):
    for (b_lo, b_hi, i_lo, i_hi) in breakpoints:
        if b_lo <= Cp <= b_hi:
            return i_lo + (Cp - b_lo) * (i_hi - i_lo) / (b_hi - b_lo)
    # If Cp > highest breakpoint:
    b_lo, b_hi, i_lo, i_hi = breakpoints[-1]
    if Cp > b_hi:
        # linear extrapolation using the last interval slope
        return i_lo + (Cp - b_lo) * (i_hi - i_lo) / (b_hi - b_lo)
    return np.nan

# Breakpoints commonly used in rohanrao/air-quality-data-in-india:
# Let's check PM2.5, PM10, CO breakpoints
# PM2.5: (0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300), (120, 250, 300, 400), (250, 400, 400, 500)
# PM10: (0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300), (350, 430, 300, 400), (430, 500, 400, 500)
# CO: (0, 1, 0, 50), (1, 2, 50, 100), (2, 10, 100, 200), (10, 17, 200, 300), (17, 34, 300, 400), (34, 40, 400, 500) Wait or (34, 50)?

bp_pm25 = [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300), (120, 250, 300, 400), (250, 400, 400, 500)]
bp_pm10 = [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300), (350, 430, 300, 400), (430, 500, 400, 500)]
# Let's test CO with 34 to 40 or 34 to 50:
for co_max in [40, 45, 50, 34]:
    # if last is (17, 34, 300, 400), what is severe?
    # CPCB 2014 report says: CO 8-hr: Good 0-1, Satisfactory 1.1-2.0, Moderate 2.1-10, Poor 10.1-17, Very Poor 17.1-34, Severe: > 34
    pass

# Test DL002 on 2016-11-07: PM2.5 = 559.74, PM10 = 638.79, AQI = 791.0
pm25_val = 559.74
pm10_val = 638.79
# For PM2.5: 400 + (559.74 - 250) * 100 / (400 - 250) = 400 + 309.74 * (100/150) = 400 + 206.49 = 606.49
# Wait! If last interval is (250, 380, 400, 500) -> 400 + 309.74 * (100/130) = 400 + 238.26 = 638.26
# What about PM10? If PM10 is (430, 500, 400, 500): slope is 100 / 70 = 1.42857!
# 400 + (638.79 - 430) * (100 / 70) = 400 + 208.79 * 1.42857 = 400 + 298.27 = 698.27
# Wait, what if PM10 is (430, ?, 400, 500)?
# Let's test what function yields 791.0!

# Let's check GJ001 on 2018-02-19: CO = 132.47, AQI = 2049.0
# Notice: AQI is 2049.0!
# If CO subindex = 400 + (132.47 - 34) * slope = 2049!
# (132.47 - 34) = 98.47.
# (2049 - 400) = 1649.
# 1649 / 98.47 = 16.746!
# Where does slope 16.746 come from?
# If the last bracket for CO was: 17 to 34 is 300 to 400 (range 17, AQI range 100 -> slope = 100/17 = 5.88).
# What if Severe was 34 to 40 -> AQI range 100 (400 to 500) -> range 6 -> slope = 100/6 = 16.6667!
# Let's compute: 400 + (132.47 - 34) * (100 / 6) = 400 + 98.47 * 16.66667 = 400 + 1641.16 = 2041.16!
# Or if B_lo=34, B_hi=?, let's test!
print("400 + (132.47 - 34) * (100 / 6) =", 400 + (132.47 - 34) * (100 / 6))

