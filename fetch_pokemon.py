import requests
import time
import os

GENERATIONS = {
    "gen1": (1, 151),
    "gen2": (152, 251),
    "gen3": (252, 386),
    "gen4": (387, 493),
    "gen5": (494, 649),
    "gen6": (650, 721),
    "gen7": (722, 809),
    "gen8": (810, 905),
    "gen9": (906, 1025),
}

def fetch_pokemon(name_or_id):
    url = f"https://pokeapi.co/api/v2/pokemon/{name_or_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def pokemon_to_text(data):
    name = data["name"].capitalize()
    types = ", ".join([t["type"]["name"] for t in data["types"]])
    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
    abilities = ", ".join([a["ability"]["name"] for a in data["abilities"]])
    moves = ", ".join([m["move"]["name"] for m in data["moves"]][:20])

    text = f"""--- {name} ---
Nome: {name}
Tipi: {types}
Abilita: {abilities}
Mosse: {moves}
HP: {stats.get("hp", "?")}
Attacco: {stats.get("attack", "?")}
Difesa: {stats.get("defense", "?")}
Attacco Speciale: {stats.get("special-attack", "?")}
Difesa Speciale: {stats.get("special-defense", "?")}
Velocita: {stats.get("speed", "?")}

"""
    return text

def fetch_all_pokemon():
    os.makedirs("data", exist_ok=True)

    for gen_name, (start, end) in GENERATIONS.items():
        print(f"\nScaricando {gen_name} (#{start}-#{end})...")
        gen_text = ""

        for i in range(start, end + 1):
            print(f"  Pokemon {i}/{end}...", end="\r")
            data = fetch_pokemon(i)
            if data:
                gen_text += pokemon_to_text(data)
            time.sleep(0.1)

        filename = f"data/{gen_name}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(gen_text)
        print(f"  {gen_name}.txt salvato!")

    print("\nDownload completato!")

if __name__ == "__main__":
    fetch_all_pokemon()