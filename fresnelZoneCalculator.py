#!/usr/bin/python3

from math import sqrt


def semiMinorAxis(losLength, nLosLength):
    return sqrt(nLosLength**2 - losLength**2) / 2


distanceBetweenFoci = input("Distance between foci? (m) ")
freq = input("Frequency? (2437 MHz) ")
speedOfWave = input("Speed of wave? (c) ")

distanceBetweenFoci = float(distanceBetweenFoci)

if freq == "":
    freq = 2437000000
else:
    freq = float(freq) * 1000000

if speedOfWave == "":
    speedOfWave = 299792458
else:
    speedOfWave = float(speedOfWave)

print("")

wavelength = speedOfWave / freq
print(f"Wavelength: {round(wavelength * 10000.0) / 100.0} cm")

losLength = distanceBetweenFoci
for i in range(0, 7):
    print(f"{round(semiMinorAxis(losLength, losLength + (wavelength / 2.0) * (2*i)) * 100.0) / 100.0} cm (constructive)")
    print(f"{round(semiMinorAxis(losLength, losLength + (wavelength / 2.0) * (2*i + 1)) * 100.0) / 100.0} cm (destructive)")

print("")
