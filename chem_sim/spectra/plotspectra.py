import re
import matplotlib.pyplot as plt
import numpy as np

data = """
##JCAMPDX=4.24
##DATA TYPE=INFRARED SPECTRUM
##ORIGIN=EPAIR VAPOR PHASE LIBRARY
##OWNER=NIST OSRD
##CAS REGISTRY NO=99081
##MOLFORM=C 7 H 7 N O 2
##DELTAX=6.000000
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##XFACTOR=1.000000
##YFACTOR=1.000000
##FIRSTX=500.000000
##LASTX=3980.000000
##NPOINTS=580
##XYDATA=(X++(Y..Y))
500.000 53 50 35 40 41 31 30 37 34 34
560.000 28 31 34 25 30 28 25 30 30 33
620.000 30 29 32 31 49 76 137 250 378 407
680.000 319 160 86 63 102 260 542 760 1423 456
740.000 183 80 54 57 76 107 191 397 1047 1417
800.000 1105 557 125 49 47 53 54 50 48 53
860.000 63 91 124 156 156 185 216 221 195 140
920.000 128 83 65 48 41 43 39 37 36 39
980.000 38 39 43 44 44 43 38 39 47 57
1040.000 75 84 99 123 173 224 246 297 338 348
1100.000 272 154 74 43 38 41 46 45 47 52
1160.000 50 48 41 39 43 46 49 55 56 54
1220.000 48 47 45 45 48 56 69 94 132 187
1280.000 260 298 308 265 229 210 244 338 567 1024
1340.000 1863 3168 3631 3082 1462 570 319 239 211 169 
1400.000 142 124 115 106 115 136 149 163 163 165
1460.000 168 222 329 392 425 401 360 357 415 625
1520.000 1176 2231 3533 3992 3961 3047 2113 1610 1146 775 
1580.000 619 540 451 340 245 174 129 98 88 67
1640.000 68 69 60 57 54 54 52 37 38 30
1700.000 31 45 51 52 47 31 28 33 34 40
1760.000 44 45 43 45 45 51 60 55 46 32
1820.000 22 18 20 24 29 34 35 29 22 24
1880.000 29 45 48 42 28 16 16 17 20 30
1940.000 39 41 40 29 18 14 14 14 15 13
2000.000 15 12 10 9 10 6 8 7 7 9
2060.000 11 8 9 9 10 6 9 9 6 9
2120.000 9 8 11 8 8 11 12 12 13 13
2180.000 12 11 10 7 8 8 8 7 6 5
2240.000 9 11 14 12 13 12 10 14 16 18
2300.000 19 19 20 26 28 27 25 25 23 20 
2360.000 16 11 14 17 15 14 12 8 9 8 
2420.000 10 14 16 21 21 21 18 14 10 7 
2480.000 5 6 6 5 2 3 1 0 1 0
2540.000 1 0 3 5 11 16 16 15 13 14
2600.000 14 11 11 9 8 8 9 12 9 9
2660.000 11 11 14 17 16 20 21 19 20 15
2720.000 12 8 12 17 22 25 17 14 14 11
2780.000 17 20 23 22 23 24 27 32 37 44
2840.000 51 59 74 102 138 175 195 193 177 164 
2900.000 163 173 194 243 312 350 342 292 231 179 
2960.000 144 119 101 94 103 119 138 154 164 163 
3020.000 180 218 248 243 199 153 150 192 240 246 
3080.000 222 148 113 83 61 38 24 16 8 10
3140.000 5 8 4 8 8 5 6 7 6 7
3200.000 10 10 7 8 8 11 8 10 7 9 
3260.000 11 9 6 7 8 10 10 11 10 9 
3320.000 9 11 7 10 11 9 10 7 7 11 
3380.000 7 7 11 8 8 11 9 11 11 9 
3440.000 9 12 13 10 8 8 10 10 9 9 
3500.000 10 8 11 11 5 9 10 12 13 11 
3560.000 11 10 9 11 12 8 13 14 9 10 
3620.000 14 11 10 11 0 12 12 15 14 9 
3680.000 13 10 13 16 14 16 16 16 16 10 
3740.000 2 3 9 16 15 17 15 12 12 7 
3800.000 7 11 8 10 9 11 10 4 8 5 
3860.000 4 3 4 9 11 13 11 8 10 10 
3920.000 9 8 7 7 9 15 12 10 12 12 
##END
"""

# 从data中提取有效部分
data_lines = data.split('\n')
start_data = False
x_data, y_data = [], []
x_waves = []

# 提取信息
x_units = re.search(r'##XUNITS=(.*)', data).group(1)
y_units = re.search(r'##YUNITS=(.*)', data).group(1)
molform = re.search(r'##MOLFORM=(.*)', data).group(1)
delta_x = float(re.search(r"##DELTAX=(.*)", data).group(1))

for line in data_lines:
    if line.startswith("##XYDATA"):
        start_data = True
        continue
    elif line.startswith("##END"):
        break

    if start_data:
        values = list(map(float, line.split()))
        x_value = values[0]
        x_values = np.arange(x_value, x_value + delta_x * len(values[1:]), delta_x)
        x_wave = [1/x*10000 for x in x_values]
        y_values = np.array(values[1:])
        x_data.extend(x_values)
        y_data.extend(y_values)
        x_waves.extend(x_wave)

# 绘制红外光谱图
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, label='Infrared Spectrum', color='blue')
plt.xlabel(f'Wavenumber ({x_units})')
plt.ylabel(f'{y_units}')
plt.title(f'Infrared Spectrum of: {molform}')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_waves, y_data, label='Infrared Spectrum', color='blue')
plt.xlabel(f'Wavelength (um)')
plt.ylabel(f'{y_units}')
# plt.xlim([0.4,0.8])
plt.title(f'Infrared Spectrum of: {molform}')
plt.legend()
plt.grid(True)
plt.show()
print(x_data)
print(x_waves)
print(y_data)