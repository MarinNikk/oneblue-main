# -*- coding: utf-8 -*-
"""ROS 2 čvor (node) koji objavljuje polje morskih struja Jadrana na dvije standardne teme.

  /ocean/currents         nav_msgs/OccupancyGrid   (iznos brzine kao toplinska karta)
  /ocean/current_vectors  geometry_msgs/PoseArray  (strelice smjera struje)

Tema (topic) je imenovani kanal na koji čvor objavljuje poruke, a drugi ih slušaju.
Koristimo standardne tipove poruka (OccupancyGrid i PoseArray) pa ih RViz2 crta bez
ikakvih vlastitih .msg definicija. Sami podaci dolaze iz field_source.py, koji je
namjerno ROS-neovisan (vidi objašnjenje ondje o dva Python okruženja).

Pokretanje:  python3 current_publisher.py            (ili: ros2 run ...)
Vizualizacija:  rviz2  -> postavi Fixed Frame na "map", pa dodaj Map (/ocean/currents)
                i PoseArray (/ocean/current_vectors).
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, TransformStamped
from tf2_ros import StaticTransformBroadcaster

from field_source import (load_field, grid_geometry, speed_to_occupancy,
                          current_vectors)

FRAME = "map"   # naziv koordinatnog okvira (frame) u kojem su sve poruke; RViz ga traži kao Fixed Frame


def yaw_to_quat(yaw):
    """Pretvori kut yaw (rotacija oko okomite z-osi, u radijanima) u kvaternion.

    ROS orijentacije ne zapisuje kao kut nego kao kvaternion (x, y, z, w). Kako se
    strelica okreće samo u ravnini (oko z-osi), x i y su 0, a z i w slijede iz
    poznate formule za rotaciju oko jedne osi (sin i cos polovice kuta).
    """
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


class CurrentPublisher(Node):
    """ROS 2 čvor koji u pravilnim razmacima objavljuje trenutno polje struja.

    Pri pokretanju stvara dva objavljivača (publisher), emitira statični TF i pokreće
    tajmer koji poziva tick(). Svaki tick objavi jedno dnevno polje na obje teme.
    """

    def __init__(self):
        super().__init__("ocean_current_publisher")
        # Parametri se mogu mijenjati izvana pri pokretanju bez diranja koda.
        self.declare_parameter("stride", 3)          # prorjeđivanje mreže pri čitanju polja
        self.declare_parameter("subsample", 2)       # gustoća strelica (svaka druga ćelija)
        self.declare_parameter("period_s", 1.0)      # razmak između objava, u sekundama (animacija po danima)
        # create_publisher(tip_poruke, naziv_teme, dubina_reda). Broj 10 je veličina reda čekanja.
        self.grid_pub = self.create_publisher(OccupancyGrid, "/ocean/currents", 10)
        self.vec_pub = self.create_publisher(PoseArray, "/ocean/current_vectors", 10)
        self._publish_static_tf()      # da okvir "map" postoji u TF stablu (inače RViz nema Fixed Frame)
        self.t = 0                     # indeks trenutnog dana/polja koje objavljujemo
        period = self.get_parameter("period_s").value
        self.timer = self.create_timer(period, self.tick)   # tajmer periodično zove tick()
        self.get_logger().info("objavljujem na /ocean/currents i /ocean/current_vectors")

    def _publish_static_tf(self):
        """Objavi statični identitetski TF iz okvira map u okvir ocean.

        TF (transform) opisuje odnos dvaju koordinatnih okvira. RViz traži Fixed Frame
        (ovdje "map"), a taj okvir mora postojati u TF stablu inače se prikaz ne iscrtava.
        Transformacija je identitet (bez pomaka, rotacija w=1.0), tj. map i ocean se
        poklapaju; ovo je samo da stablo okvira postoji. Statični TF se objavi jednom.
        """
        self._tf = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = FRAME
        tf.child_frame_id = "ocean"
        tf.transform.rotation.w = 1.0   # w=1, ostalo 0 => rotacija je identitet (bez zakreta)
        self._tf.sendTransform(tf)

    def tick(self):
        """Poziva ga tajmer: učitaj polje za dan self.t i objavi ga na obje teme.

        Kad dođemo do kraja niza dana, load_field baci IndexError; tada se vraćamo na
        prvi dan (t=0) pa animacija kreće ispočetka, kao petlja.
        """
        stride = self.get_parameter("stride").value
        sub = self.get_parameter("subsample").value
        try:
            f = load_field(t_index=self.t, stride=stride)
        except IndexError:                 # nema više dana: vrti animaciju od početka
            self.t = 0
            f = load_field(t_index=0, stride=stride)
        stamp = self.get_clock().now().to_msg()     # zajednička vremenska oznaka za obje poruke
        self.grid_pub.publish(self._grid_msg(f, stamp))
        self.vec_pub.publish(self._vec_msg(f, sub, stamp))
        self.get_logger().info(f"objavljeno polje za {f['time']}")
        self.t += 1

    def _grid_msg(self, f, stamp):
        """Od polja f napravi nav_msgs/OccupancyGrid poruku (toplinska karta iznosa brzine)."""
        g = grid_geometry(f["lat"], f["lon"])
        occ, _ = speed_to_occupancy(f["u"], f["v"])
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME
        msg.info.resolution = g["resolution"]
        msg.info.width = g["width"]
        msg.info.height = g["height"]
        msg.info.origin.position = Point(x=0.0, y=0.0, z=0.0)   # ishodište grida u 0,0
        # OccupancyGrid.data je jednodimenzionalna lista; flatten(order="C") slaže
        # ćelije redak po redak (row-major), počevši od juga, kako poruka očekuje.
        msg.data = occ.flatten(order="C").tolist()
        return msg

    def _vec_msg(self, f, sub, stamp):
        """Od polja f napravi geometry_msgs/PoseArray poruku (strelice smjera struje)."""
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME
        # Za svaku strelicu iz jezgre dodaj jedan Pose: položaj (x, y) i orijentaciju (yaw).
        for x, y, yaw, _ in current_vectors(f["lat"], f["lon"], f["u"], f["v"], sub):
            msg.poses.append(Pose(position=Point(x=x, y=y, z=0.0),
                                  orientation=yaw_to_quat(yaw)))
        return msg


def main(args=None):
    """Ulazna točka: pokreni ROS, vrti čvor dok se ne prekine, pa čisto ugasi.

    rclpy.spin drži čvor živim i obrađuje tajmere/poruke. Kod prekida (Ctrl+C ili
    vanjsko gašenje) hvatamo iznimke da se program ne sruši s tragom greške. U finally
    uredno gasimo: uništimo čvor, a rclpy.shutdown zovemo samo ako je ROS još aktivan
    (if rclpy.ok()), da ne zovemo shutdown dvaput.
    """
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = CurrentPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
