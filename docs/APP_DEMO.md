# App Demo: Paüls Wildfire (Catalonia, July 2025)

Demo of the Fire Detection App using a well-known wildfire in southern Catalonia.

---

## Executive Summary

- **Demo:** Fire Detection App tested on the Paüls wildfire (southern Catalonia, July 2025)
- **Event:** ~3,200 ha burned; fire crossed the Ebre river; 18,000 confined
- **Region:** Paüls / Baix Ebre (preset in app)
- **Pre-fire (June 2025):** App should detect no fire
- **Post-fire (July 8+):** App should detect fire and severity maps
- **Success:** App correctly distinguishes pre- vs post-fire imagery and produces plausible burn maps

---

## Fire Event — Key Facts

| Field | Value |
|-------|-------|
| **Location** | Paüls, Baix Ebre county, southern Catalonia |
| **Coordinates** | ~40.67°N, 0.40°E (Paüls area) |
| **Start date** | Monday, July 7, 2025 (~12:00) |
| **Peak extent** | ~3,200 hectares burned |
| **Land use** | ~2,300 ha forest, ~820 ha agricultural, ~25 ha urban |
| **Area affected** | Els Ports natural park, Ebre river crossed |
| **Towns** | Paüls, Aldover, Xerta, Alfara de Carles, Jesús, Bitem, Roquetes, Tivenys, Tortosa (Reguers), Prat del Comte, Pinell de Brai |
| **Evacuations** | 50+ residents; 18,000 confined at peak |

**Sources:**
- [Wildfire in southern Catalonia escalates (July 7)](https://www.catalannews.com/society-science/item/wildfire-pauls-southern-catalonia-july-7)
- [Firefighters see 'favorable progress' in Paüls wildfire (July 8)](https://www.catalannews.com/society-science/item/wildfire-pauls-southern-catalonia-8-july-2025)

---

## Demo Setup

**Region:** Paüls / Baix Ebre (preset in app or bbox around 40.67°N, 0.40°E)

**Assumptions:**
- **Before fire (e.g. June 2025):** App should detect **no fire** — imagery shows pre-fire vegetation.
- **After fire (e.g. July 8–15, 2025):** App should detect **fire** and **severity** — burn scars visible in post-fire imagery.

---

## Expected Outcome

- **Pre-fire run:** No fire detected (or negligible).
- **Post-fire run:** Fire detected; binary map and 5-level severity map show burned area consistent with the event.

**Conclusion:** Demo succeeds if the app correctly distinguishes pre-fire vs post-fire imagery and produces plausible fire and severity maps for the Paüls event.
