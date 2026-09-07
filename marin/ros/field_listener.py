# -*- coding: utf-8 -*-
"""ROS 2 pretplatnik (subscriber): sluša obje teme i zapiše sažetak u tested/summary.json.

Ovo je mali alat za provjeru da objavljivanje radi, bez pokretanja RViz-a. Pretplati
se na iste teme na koje current_publisher.py objavljuje, skupi zadani broj poruka,
zapiše kratki sažetak (dimenzije grida, broj morskih ćelija, broj strelica ...) i izađe.
Pokretanje (dok current_publisher već radi):  python3 field_listener.py
"""
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray

OUT = Path(__file__).resolve().parent / "tested"   # mapa u koju spremamo summary.json


class Listener(Node):
    """Čvor koji broji primljene poruke s obje teme i na kraju zapiše sažetak.

    target je koliko poruka sa svake teme treba primiti prije nego se posao smatra
    gotovim. Callback funkcije (on_grid, on_vec) ROS poziva svaki put kad stigne poruka.
    """

    def __init__(self, target=10):
        super().__init__("ocean_current_listener")
        self.target = target
        self.grid_msgs = 0        # brojač primljenih OccupancyGrid poruka
        self.vec_msgs = 0         # brojač primljenih PoseArray poruka
        self.last = {}            # zadnje viđene vrijednosti, idu u sažetak
        # create_subscription(tip_poruke, naziv_teme, callback, dubina_reda):
        # registrira funkciju koju ROS zove kad na toj temi stigne nova poruka.
        self.create_subscription(OccupancyGrid, "/ocean/currents", self.on_grid, 10)
        self.create_subscription(PoseArray, "/ocean/current_vectors", self.on_vec, 10)
        self.get_logger().info("slušam /ocean/currents i /ocean/current_vectors ...")

    def on_grid(self, m):
        """Callback za /ocean/currents: zabilježi dimenzije grida i raspon vrijednosti."""
        self.grid_msgs += 1
        sea = [d for d in m.data if d >= 0]    # >= 0 su morske ćelije (kopno je -1)
        self.last["grid"] = dict(width=m.info.width, height=m.info.height,
                                 resolution_m=round(m.info.resolution, 1),
                                 sea_cells=len(sea), land_cells=len(m.data) - len(sea),
                                 occ_min=min(sea) if sea else None,
                                 occ_max=max(sea) if sea else None,
                                 frame=m.header.frame_id)
        self._maybe_done()

    def on_vec(self, m):
        """Callback za /ocean/current_vectors: zabilježi broj strelica (poses)."""
        self.vec_msgs += 1
        self.last["vectors"] = dict(arrows=len(m.poses), frame=m.header.frame_id)
        self._maybe_done()

    def _maybe_done(self):
        """Kad primimo dovoljno poruka s obje teme, zapiši sažetak i izađi.

        Čeka da oba brojača dosegnu target (da imamo i grid i strelice). SystemExit
        prekida rclpy.spin, pa ga main() uhvati i čvor se uredno ugasi.
        """
        if self.grid_msgs >= self.target and self.vec_msgs >= self.target:
            OUT.mkdir(exist_ok=True)
            summary = dict(status="OK", grid_msgs=self.grid_msgs,
                           vec_msgs=self.vec_msgs, **self.last)
            (OUT / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False))   # ensure_ascii=False: čuva čć đ š ž
            self.get_logger().info(f"sažetak zapisan: {OUT/'summary.json'}")
            raise SystemExit


def main():
    """Ulazna točka: pokreni ROS, slušaj dok ne skupiš dovoljno poruka, pa čisto ugasi.

    Isti obrazac urednog gašenja kao u current_publisher.py: hvatamo iznimke (uključujući
    SystemExit iz _maybe_done) i u finally uništimo čvor, a rclpy.shutdown zovemo samo
    ako je ROS još aktivan.
    """
    from rclpy.executors import ExternalShutdownException
    rclpy.init()
    node = Listener()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
