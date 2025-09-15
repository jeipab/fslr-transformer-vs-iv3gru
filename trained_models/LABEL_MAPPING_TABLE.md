# Label Mapping Table

This document provides a complete mapping of class IDs to human-readable labels for the Sign Language Recognition system.

## Categories (10 total)

| ID  | Category      | Description                          |
| --- | ------------- | ------------------------------------ |
| 0   | GREETING      | Greetings and polite expressions     |
| 1   | SURVIVAL      | Basic communication needs            |
| 2   | NUMBER        | Numbers 1-10                         |
| 3   | CALENDAR      | Months of the year                   |
| 4   | DAYS          | Days of the week and time references |
| 5   | FAMILY        | Family relationships                 |
| 6   | RELATIONSHIPS | People and relationships             |
| 7   | COLOR         | Colors and visual descriptors        |
| 8   | FOOD          | Food items                           |
| 9   | DRINK         | Beverages and drinks                 |

## Gloss Labels (105 total)

### GREETING (Category 0)

| ID  | Gloss            |
| --- | ---------------- |
| 0   | GOOD MORNING     |
| 1   | GOOD AFTERNOON   |
| 2   | GOOD EVENING     |
| 3   | HELLO            |
| 4   | HOW ARE YOU      |
| 5   | IM FINE          |
| 6   | NICE TO MEET YOU |
| 7   | THANK YOU        |
| 8   | YOURE WELCOME    |
| 9   | SEE YOU TOMORROW |

### SURVIVAL (Category 1)

| ID  | Gloss            |
| --- | ---------------- |
| 10  | UNDERSTAND       |
| 11  | DON'T UNDERSTAND |
| 12  | KNOW             |
| 13  | DON'T KNOW       |
| 14  | NO               |
| 15  | YES              |
| 16  | WRONG            |
| 17  | CORRECT          |
| 18  | SLOW             |
| 19  | FAST             |

### NUMBER (Category 2)

| ID  | Gloss |
| --- | ----- |
| 20  | ONE   |
| 21  | TWO   |
| 22  | THREE |
| 23  | FOUR  |
| 24  | FIVE  |
| 25  | SIX   |
| 26  | SEVEN |
| 27  | EIGHT |
| 28  | NINE  |
| 29  | TEN   |

### CALENDAR (Category 3)

| ID  | Gloss     |
| --- | --------- |
| 30  | JANUARY   |
| 31  | FEBRUARY  |
| 32  | MARCH     |
| 33  | APRIL     |
| 34  | MAY       |
| 35  | JUNE      |
| 36  | JULY      |
| 37  | AUGUST    |
| 38  | SEPTEMBER |
| 39  | OCTOBER   |
| 40  | NOVEMBER  |
| 41  | DECEMBER  |

### DAYS (Category 4)

| ID  | Gloss     |
| --- | --------- |
| 42  | MONDAY    |
| 43  | TUESDAY   |
| 44  | WEDNESDAY |
| 45  | THURSDAY  |
| 46  | FRIDAY    |
| 47  | SATURDAY  |
| 48  | SUNDAY    |
| 49  | TODAY     |
| 50  | TOMORROW  |
| 51  | YESTERDAY |

### FAMILY (Category 5)

| ID  | Gloss       |
| --- | ----------- |
| 52  | FATHER      |
| 53  | MOTHER      |
| 54  | SON         |
| 55  | DAUGHTER    |
| 56  | GRANDFATHER |
| 57  | GRANDMOTHER |
| 58  | UNCLE       |
| 59  | AUNTIE      |
| 60  | COUSIN      |
| 61  | PARENTS     |

### RELATIONSHIPS (Category 6)

| ID  | Gloss            |
| --- | ---------------- |
| 62  | BOY              |
| 63  | GIRL             |
| 64  | MAN              |
| 65  | WOMAN            |
| 66  | DEAF             |
| 67  | HARD OF HEARING  |
| 68  | WEELCHAIR PERSON |
| 69  | BLIND            |
| 70  | DEAF BLIND       |
| 71  | MARRIED          |

### COLOR (Category 7)

| ID  | Gloss  |
| --- | ------ |
| 72  | BLUE   |
| 73  | GREEN  |
| 74  | RED    |
| 75  | BROWN  |
| 76  | BLACK  |
| 77  | WHITE  |
| 78  | YELLOW |
| 79  | ORANGE |
| 80  | GRAY   |
| 81  | PINK   |
| 82  | VIOLET |
| 83  | LIGHT  |
| 84  | DARK   |

### FOOD (Category 8)

| ID  | Gloss     |
| --- | --------- |
| 85  | BREAD     |
| 86  | EGG       |
| 87  | FISH      |
| 88  | MEAT      |
| 89  | CHICKEN   |
| 90  | SPAGHETTI |
| 91  | RICE      |
| 92  | LONGANISA |
| 93  | SHRIMP    |
| 94  | CRAB      |

### DRINK (Category 9)

| ID  | Gloss    |
| --- | -------- |
| 95  | HOT      |
| 96  | COLD     |
| 97  | JUICE    |
| 98  | MILK     |
| 99  | COFFEE   |
| 100 | TEA      |
| 101 | BEER     |
| 102 | WINE     |
| 103 | SUGAR    |
| 104 | NO SUGAR |

## Example Prediction Results

Based on your test prediction of "clip_0089_how are you.npz":

**Predicted Gloss:** HOW ARE YOU (ID: 4, confidence: 88.2%)  
**Predicted Category:** GREETING (ID: 0, confidence: 77.4%)

**Top 5 Gloss Predictions:**

1. HOW ARE YOU (ID: 4): 88.2%
2. SLOW (ID: 18): 7.4%
3. CORRECT (ID: 17): 1.3%
4. BREAD (ID: 85): 0.7%
5. NICE TO MEET YOU (ID: 6): 0.6%

**Top 3 Category Predictions:**

1. GREETING (ID: 0): 77.4%
2. FOOD (ID: 8): 16.0%
3. SURVIVAL (ID: 1): 6.1%

The model correctly identified "HOW ARE YOU" as a GREETING with high confidence!
