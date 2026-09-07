# Objava polja struja na ROS 2

Minimalna integracija s Robotskim operacijskim sustavom (ROS 2): jedan čvor koji polje
morskih struja (izmjereno, interpolirano ili **predviđeno**) objavljuje na dvije
**standardne** teme, pa se odmah prikazuje u RViz2 bez ijednog vlastitog tipa poruke.

## Teme
| Tema | Tip poruke | Sadržaj |
|---|---|---|
| `/ocean/currents` | `nav_msgs/OccupancyGrid` | brzina struje kao toplinska karta (0-100, kopno = -1) |
| `/ocean/current_vectors` | `geometry_msgs/PoseArray` | strelice smjera struje po morskim ćelijama |

Izvor polja je namjerno odvojen od objave (isti čvor prima i predikciju modela),
u skladu s modulom OneDrift.

## Datoteke
| Datoteka | Uloga |
|---|---|
| `field_source.py` | učitavanje polja + pretvorba u grid/vektore (**bez ROS-a**, samo numpy) |
| `precompute_fields.py` | NetCDF -> `cache/fields.npz` (pokreni iz `.venv`; da ROS ne treba xarray) |
| `current_publisher.py` | ROS 2 čvor (`rclpy`): objavljuje na dvije teme, animacija po danima |
| `field_listener.py` | pretplatnik koji sažetak zapiše u `tested/summary.json` (dokaz da objava radi) |
| `run_test.sh` | jednokratni test: izdavač + rosbag + sažetak |
| `install_ros2_jazzy.sh` | instalacija ROS 2 Jazzy + ovisnosti (Ubuntu 24.04) |

## Okruženje
Ubuntu 24.04 -> **ROS 2 Jazzy**. ROS koristi sistemski Python 3.12, a `.venv`
(s xarray-em) je 3.13, pa polja unaprijed pretvaramo u `cache/fields.npz`
(numpy). Predmemorija je već izrađena; ROS čvor treba samo `python3-numpy`.

## Pokretanje
```bash
# 0) (već napravljeno) predmemorija iz .venv-a
../../.venv/bin/python precompute_fields.py

# 1) instalacija ROS-a (jednom, traži lozinku)
sudo bash install_ros2_jazzy.sh

# 2) test i snimanje podataka
source /opt/ros/jazzy/setup.bash
bash run_test.sh                 # -> tested/summary.json + tested/adriatic_bag/

# 3) vizualizacija
source /opt/ros/jazzy/setup.bash
python3 current_publisher.py &
rviz2
#   Fixed Frame: map
#   Add -> Map        -> tema /ocean/currents        (toplinska karta brzine)
#   Add -> PoseArray  -> tema /ocean/current_vectors (strelice smjera)
```
Čvor na svakom koraku objavljuje polje sljedećega dana, pa se u RViz2 vidi
animacija kretanja struja. Snimljeni rosbag i `summary.json` ostaju kao dokaz da
objava radi od kraja do kraja.

## Parametri
| Parametar | Zadano | Značenje |
|---|---|---|
| `stride` | 3 | prorjeđivanje mreže |
| `subsample` | 2 | gustoća strelica |
| `period_s` | 1.0 | razmak objava (s) |

```bash
python3 current_publisher.py --ros-args -p period_s:=0.5 -p subsample:=1
```

## Zašto standardne teme
`OccupancyGrid` za skalarnu veličinu i standardni tip s orijentacijom za
vektorsko polje, uz vremensku oznaku i koordinatni okvir u zaglavlju. Objava na
teme omogućuje da isto polje istodobno koriste vizualizacija, simulator zanošenja
i upravljanje plovilom.

## Napomena
Radi jednostavnosti polje je smješteno u lokalni `map` okvir (metri, ishodište u
jugozapadnom kutu; stupnjevi su pretvoreni u metre uz kosinusnu korekciju širine).
Za operativnu uporabu prirodno je dodati geografski (UTM/ENU) okvir i `tf`
transformaciju.
