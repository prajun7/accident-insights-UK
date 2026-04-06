# Feature Selection: V1 vs V2 — What Changed and Why

## Overview

| | V1 (Original) | V2 (Updated) |
|-|--------------|--------------|
| Input columns | 12 (11 features + target) | 22 (21 features + target) |
| Feature categories | Road conditions only | Road + Vehicle + Driver |
| LDA selects top | 10 | 10 |
| Classification F1-Macro | 0.4087 (RF) | TBD after re-run |

---

## Columns Compared

### Kept from V1 (unchanged)
| Column | Category | Reason kept |
|--------|----------|-------------|
| `Speed_limit` | Road | Strongest LDA coefficient in V1 |
| `Road_Type` | Road | High LD1 coefficient |
| `Light_Conditions` | Road | Visibility directly affects severity |
| `Weather_Conditions` | Road | Rain/fog increases crash severity |
| `Road_Surface_Conditions` | Road | Wet/icy surface affects outcomes |
| `Urban_or_Rural_Area` | Road | Rural crashes tend to be more severe |
| `Junction_Detail` | Road | Junction type affects crash type |
| `Junction_Control` | Road | Traffic signals vs uncontrolled |
| `Number_of_Vehicles` | Accident | Multi-vehicle crashes differ from single |
| `Hour` | Time | Visibility and traffic density by hour |
| `IsNight` | Time | Night crashes have different severity profiles |

---

### Removed from V1
| Column | Reason removed |
|--------|---------------|
| *(none removed)* | All V1 features carried forward into V2 |

---

### Added in V2
| Column | Category | Why added |
|--------|----------|-----------|
| `Vehicle_Type` | Vehicle | Motorcycle vs car vs truck — crash with a truck is fundamentally more severe than a car |
| `Vehicle_Manoeuvre` | Vehicle | What the vehicle was doing before impact (turning, reversing, overtaking) directly affects crash type |
| `Towing_and_Articulation` | Vehicle | Vehicles towing a trailer or articulated lorries behave differently in crashes |
| `Age_Band_of_Driver` | Driver | Young (17–24) and elderly (75+) drivers have significantly different severity outcomes |
| `Sex_of_Driver` | Driver | Available pre-crash, correlates with risk-taking behaviour |
| `Journey_Purpose_of_Driver` | Driver | Commuting vs leisure vs work — work drivers are often more fatigued |
| `1st_Road_Class` | Road | Motorway vs A-road vs B-road vs unclassified — speed and infrastructure differ significantly |
| `Special_Conditions_at_Site` | Road | Roadworks, loose chippings, oil on road — pre-existing hazards |
| `Carriageway_Hazards` | Road | Objects on road, previous accidents nearby |
| `Pedestrian_Crossing-Physical_Facilities` | Road | Zebra crossing vs pelican vs no crossing — affects pedestrian severity |

---

### Intentionally excluded (both versions)
| Column | Reason |
|--------|--------|
| `Number_of_Casualties` | **Data leakage** — casualty count is recorded at the same time as severity, not before |
| `Casualty_Severity` | **Data leakage** — directly correlated with Accident_Severity |
| `Skidding_and_Overturning` | **Data leakage** — crash outcome, not pre-condition |
| `Hit_Object_in_Carriageway` | **Data leakage** — crash outcome |
| `Vehicle_Leaving_Carriageway` | **Data leakage** — crash outcome |
| `1st_Point_of_Impact` | **Data leakage** — where vehicle was hit, known only after crash |
| `Month`, `DayOfWeek`, `IsWeekend` | **Frequency bias** — predict *when* accidents happen, not *how severe* they are |
| `Longitude`, `Latitude` | Saved separately in `lat_lon.csv` for geographic clustering |

---

## Why V1 Had Low Accuracy

V1 only used **road and environmental conditions**. These tell you the setting of the crash but not who was involved or what vehicle they were driving. Two crashes on the same road in the same weather can have very different outcomes depending on:

- Whether the vehicle was a motorcycle (more fatal) or an SUV
- Whether the driver was 19 years old or 45
- Whether they were overtaking at the time

V1 was essentially asking: *"Given this road and weather, how bad was the crash?"*
V2 asks: *"Given this road, weather, vehicle, and driver — how bad was the crash?"*

---

## LDA Feature Selection Logic (both versions)

LDA (Linear Discriminant Analysis) is used to rank features by how well they **separate the three severity classes** (Fatal / Serious / Slight). It considers all features together, not independently, so it accounts for relationships between them.

- Top 10 features by |LD1 coefficient| are kept
- LD1 explains 97.2% of between-class variance — the primary axis of separation
- Features with low LD1 coefficients are dropped (they add noise without adding discriminative power)

**V1 top 10 (by |LD1|):**
Speed_limit → Number_of_Vehicles → Road_Type → IsNight → Urban_or_Rural_Area → Junction_Detail → Light_Conditions → Road_Surface_Conditions → Hour → Weather_Conditions

**V2 top 10:** *to be filled after re-run*

---

## Expected Impact of V2

Adding driver and vehicle features should improve:
- **Fatal recall** — motorcycles and young drivers are strong signals of fatal crashes
- **Serious F1** — vehicle type and manoeuvre help distinguish Serious from Slight
- **Overall F1-Macro** — more signal, less reliance on road conditions alone
