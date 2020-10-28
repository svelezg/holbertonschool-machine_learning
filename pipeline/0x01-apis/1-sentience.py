#!/usr/bin/env python3
"""  contains the sentientPlanets function """
import requests


def sentientPlanets():
    """

    :return:
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url is not None:
        r = requests.get(url)
        results = r.json()["results"]

        for specie in results:
            if (specie["designation"] == "sentient" or
                    specie["classification"] == "sentient"):

                planet_url = specie["homeworld"]
                if planet_url is not None:
                    p = requests.get(planet_url).json()
                    planets.append(p["name"])
        url = r.json()["next"]
    return planets
