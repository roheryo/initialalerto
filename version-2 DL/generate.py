import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# 1. Setup Constants and Configuration
TOTAL_RECORDS = 30000
POSITIVE_COUNT = 20000
NEGATIVE_COUNT = 10000
OUTBREAK_THRESHOLD = 30

# Target for Positives: 50% Outbreak, 50% Non-Outbreak
POSITIVE_OUTBREAK_TARGET = 10000
POSITIVE_NON_OUTBREAK_TARGET = 10000

# Date Range
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)
DAYS_RANGE = (END_DATE - START_DATE).days

# Geography: Davao de Oro (11 Municipalities + Sample Barangays)
# Real data structure for Davao de Oro
geo_data = {
    "COMPOSTELA": ["Poblacion", "Bagongon", "Gabi", "Lagab", "Mangayon", "Maparat", "Mapaca", "Ngan", "Osmeña", "San Miguel"],
    "LAAK": ["Poblacion", "Aguinaldo", "Ampawid", "Anitap", "Bagong Silang", "Banbanon", "Binasbas", "Cebulida", "Il Papa", "Kaligutan"],
    "MABINI": ["Pindasan", "Cadunan", "Cuambog", "Del Pilar", "Golden Valley", "Pangibiran", "San Antonio", "San Roque", "Tangnanan", "Libodon"],
    "MACO": ["Poblacion", "Anibongan", "Anislagan", "Binuangan", "Bucana", "Calabcab", "Concepcion", "Dumlan", "Elizalde", "Gubatan"],
    "MARAGUSAN": ["Poblacion", "Bagong Silang", "Bahi", "Cambagang", "Coronobe", "Katipunan", "Lahi", "Langgawisan", "Mabugnao", "Magcagong"],
    "MAWAB": ["Poblacion", "Andili", "Bawani", "Concepcion", "Malinawon", "Nueva Visayas", "Nuevo Iloco", "Saosao", "Salvacion", "Tuboran"],
    "MONKAYO": ["Poblacion", "Awao", "Babag", "Banlag", "Baylo", "Casoon", "Haguimitan", "Inambatan", "Macopa", "Mamunga"],
    "MONTEVISTA": ["Poblacion", "Banagbanag", "Banglasan", "Bankerohan", "Camansi", "Camantangan", "Concepcion", "Dauman", "Lebanon", "Linoan"],
    "NABUNTURAN": ["Poblacion", "Anislagan", "Antequera", "Basak", "Bayabas", "Bukal", "Cabacungan", "Cabidianan", "Katipunan", "Libasan"],
    "NEW BATAAN": ["Poblacion", "Andap", "Bantacan", "Batinao", "Cabinuangan", "Camanlangan", "Cogonon", "Fatima", "Kapatagan", "Katipunan"],
    "PANTUKAN": ["Kingking (Pob.)", "Bongabong", "Bongbong", "Magnaga", "Matiao", "Napnapan", "Tag-Ugpo", "Tambongon", "Tibagon", "Las Arenas"]
}

# Generate a master list of "Places" (Muni + Brgy + Purok)
# We will create enough places to support the non-outbreak distribution
all_places = []
for muni, brgys in geo_data.items():
    for brgy in brgys:
        # Create 20 Puroks per Barangay to ensure enough granularity
        for p in range(1, 21):
            purok = f"Purok {p}"
            all_places.append({"muni": muni, "brgy": brgy, "purok": purok})

# Shuffle places to randomize assignment
random.shuffle(all_places)

# 2. Allocate Places for Outbreak vs Non-Outbreak
# We need distinct places for Outbreak vs Non-Outbreak to ensure counts don't mix and violate thresholds

# Outbreak Places: Need enough places to hold 10,000 cases with >= 30 each.
# Let's say we target ~50 cases per outbreak place -> 10,000 / 50 = 200 places.
num_outbreak_places = 200
outbreak_places_pool = all_places[:num_outbreak_places]
remaining_places = all_places[num_outbreak_places:]

# Non-Outbreak Places: Need enough places to hold 10,000 cases with < 30 each.
# Let's say we target ~10 cases per place -> 10,000 / 10 = 1000 places.
num_non_outbreak_places = 1500 # Use more to be safe and spread them out
non_outbreak_places_pool = remaining_places[:num_non_outbreak_places]
# The rest of the places can be used for negative cases if needed
unused_places = remaining_places[num_non_outbreak_places:]

# 3. Distribute Positive Cases
records = []

# Helper to generate random date
def get_random_date():
    random_days = random.randint(0, DAYS_RANGE)
    return START_DATE + timedelta(days=random_days)

# Helper for demog
def get_sex():
    return random.choice(['Male', 'Female'])

def get_age():
    return random.randint(1, 90)

# A. Generate Outbreak Positives (10,000)
# Distribute 10k cases across 200 places. Average 50. Range [30, 70]
cases_per_place = [30] * num_outbreak_places # Minimum 30
remaining_to_alloc = POSITIVE_OUTBREAK_TARGET - sum(cases_per_place)

# Distribute remainder randomly
for _ in range(remaining_to_alloc):
    idx = random.randint(0, num_outbreak_places - 1)
    cases_per_place[idx] += 1

for i, place in enumerate(outbreak_places_pool):
    count = cases_per_place[i]
    for _ in range(count):
        records.append({
            "lab_result": "Positive",
            "muni": place['muni'],
            "brgy": place['brgy'],
            "purok": place['purok'],
            "group": "Outbreak"
        })

# B. Generate Non-Outbreak Positives (10,000)
# Distribute 10k cases across 1500 places. Max 29 per place.
# Simple logic: Assign randomly until done, ensuring no place exceeds 29.
non_outbreak_counts = [0] * num_non_outbreak_places
cases_allocated = 0
while cases_allocated < POSITIVE_NON_OUTBREAK_TARGET:
    idx = random.randint(0, num_non_outbreak_places - 1)
    if non_outbreak_counts[idx] < 29:
        non_outbreak_counts[idx] += 1
        cases_allocated += 1

for i, count in enumerate(non_outbreak_counts):
    if count > 0:
        place = non_outbreak_places_pool[i]
        for _ in range(count):
            records.append({
                "lab_result": "Positive",
                "muni": place['muni'],
                "brgy": place['brgy'],
                "purok": place['purok'],
                "group": "Non-Outbreak"
            })

# C. Generate Negatives (10,000)
# Assign to random places (can be anywhere, outbreak status is determined by positive count usually)
# We will just pick random places from the entire pool
for _ in range(NEGATIVE_COUNT):
    place = random.choice(all_places)
    records.append({
        "lab_result": "Negative",
        "muni": place['muni'],
        "brgy": place['brgy'],
        "purok": place['purok'],
        "group": "Negative"
    })

# 4. Fill in the rest of the dataframe fields
final_data = []
for r in records:
    date_val = get_random_date()
    age_val = get_age()
    birthdate = date_val - timedelta(days=age_val*365)
    
    row = {
        "type": "influenza-like-illness",
        "create_date": date_val.strftime("%Y-%m-%d"),
        "created_at": date_val.strftime("%Y-%m-%d %H:%M:%S"),
        "sex": get_sex(),
        "birthdate": birthdate.strftime("%Y-%m-%d"),
        "AGE": age_val,
        "AGE GROUP": f"{age_val//5*5}-{(age_val//5*5)+4}", # Simple grouping
        "current_address_region": "REGION XI (DAVAO REGION)",
        "current_address_province": "DAVAO DE ORO",
        "current_address_city": r['muni'],
        "current_address_barangay": r['brgy'],
        "current_address_street": r['purok'], # Using Street field for Purok
        "lab_result": r['lab_result'],
        "outcome": random.choice(["RECOVERED", "IMPROVING", "DIED"]) if r['lab_result'] == "Positive" else "NOT APPLICABLE",
        "classification": "Confirmed" if r['lab_result'] == "Positive" else "Suspect",
        "MORBIDITY WEEK": date_val.isocalendar()[1],
        "PROVINCE": "DAVAO DE ORO"
    }
    final_data.append(row)

# Create DataFrame
df = pd.DataFrame(final_data)

# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# Save to Excel
file_path = "Davao_de_Oro_Cases_2025-Generated-2.xlsx"
df.to_excel(file_path, index=False)

print(f"File created: {file_path}")