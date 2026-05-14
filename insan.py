"""
Gelişmiş İnsan Sayma Sistemi
- YOLOv8 + ByteTrack ile ID bazlı takip (aynı kişiyi tekrar saymaz)
- 2 bölge desteği
- Temiz sınıf yapısı
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Ayarlar
# ──────────────────────────────────────────────

@dataclass
class Config:
    rtsp_url: str      = "rtsp://kullanici:sifre@ip:port/kanal"
    model_path: str    = "yolov8n.pt"
    window_width: int  = 960
    window_height: int = 540
    confidence: float  = 0.5
    person_class: int  = 0            # COCO'da 0 = insan
    tracker: str       = "bytetrack.yaml"


# ──────────────────────────────────────────────
# Bölge (ROI) Veri Sınıfı
# ──────────────────────────────────────────────

REGION_COLORS = [
    (0, 255,   0),   # Yeşil   → Bölge 1
    (0,   0, 255),   # Kırmızı → Bölge 2
]

@dataclass
class Region:
    index: int
    polygon: np.ndarray
    seen_ids: set = field(default_factory=set)  # Bölgeden geçen benzersiz ID'ler

    @property
    def color(self) -> tuple:
        return REGION_COLORS[self.index]

    @property
    def label(self) -> str:
        return f"Bolge {self.index + 1}"

    def contains(self, point: tuple) -> bool:
        """Verilen nokta bu bölge içinde mi?"""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def register(self, track_id: int) -> None:
        """Takip ID'sini bu bölgeye kaydet."""
        self.seen_ids.add(track_id)

    @property
    def count(self) -> int:
        """Bu bölgeden geçen toplam benzersiz kişi sayısı."""
        return len(self.seen_ids)


# ──────────────────────────────────────────────
# ROI Seçici
# ──────────────────────────────────────────────

class ROISelector:
    """
    Kullanıcıdan fare tıklamalarıyla tam olarak 2 bölge noktası toplar.
    S → bölgeyi kaydet (en az 3 nokta)
    U → son noktayı geri al
    Q → çık (2 bölge seçildiyse)
    """

    NUM_REGIONS = 2

    def __init__(self, frame: np.ndarray):
        self.base_frame  = frame.copy()
        self.current_pts: list = []
        self.polygons: list    = []

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))

    def _draw(self) -> np.ndarray:
        img = self.base_frame.copy()

        # Tamamlanmış bölgeleri çiz
        for i, poly in enumerate(self.polygons):
            cv2.polylines(img, [poly], True, REGION_COLORS[i], 2)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(img, f"Bolge {i + 1}", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, REGION_COLORS[i], 2)

        # Aktif noktaları çiz
        for pt in self.current_pts:
            cv2.circle(img, pt, 5, (255, 255, 0), -1)
        if len(self.current_pts) > 1:
            cv2.polylines(img, [np.array(self.current_pts)], False, (255, 255, 0), 2)

        # Yardım metni
        region_no = len(self.polygons) + 1
        lines = [
            f"Bolge {region_no} icin noktalari tiklayin",
            "S = bolgeyi kaydet (en az 3 nokta)",
            "U = son noktayi geri al",
            "Q = secimi bitir (2 bolge secildikten sonra)",
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 20 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        return img

    def run(self) -> list:
        win = "ROI Secimi"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)

        print("ROI seçimi: S = kaydet  |  U = geri al  |  Q = bitir")

        while True:
            cv2.imshow(win, self._draw())
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if len(self.current_pts) >= 3:
                    self.polygons.append(np.array(self.current_pts, np.int32))
                    print(f"Bolge {len(self.polygons)} kaydedildi.")
                    self.current_pts = []
                    if len(self.polygons) == self.NUM_REGIONS:
                        print("2 bölge tamamlandı, takip başlıyor...")
                        break
                else:
                    print("En az 3 nokta gerekli!")

            elif key == ord('u') and self.current_pts:
                self.current_pts.pop()

            elif key == ord('q'):
                if len(self.polygons) < self.NUM_REGIONS:
                    print(f"Lütfen {self.NUM_REGIONS} bölge seçin! "
                          f"(Şu an: {len(self.polygons)})")
                else:
                    break

        cv2.destroyWindow(win)
        return self.polygons


# ──────────────────────────────────────────────
# Ana Uygulama
# ──────────────────────────────────────────────

class PeopleCounter:
    """
    RTSP akışından 2 bölge bazlı, ID takipli insan sayar.
    Aynı kişi bölgeye tekrar girse bile tekrar sayılmaz (seen_ids seti).
    """

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self.model = YOLO(cfg.model_path)
        self.cap   = cv2.VideoCapture(cfg.rtsp_url)
        self.regions: list[Region] = []

        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı: {cfg.rtsp_url}")

    # ── Kare okuma ────────────────────────────
    def _read_frame(self) -> tuple:
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        frame = cv2.resize(frame, (self.cfg.window_width, self.cfg.window_height))
        return True, frame

    # ── ROI kurulumu ──────────────────────────
    def setup_regions(self) -> None:
        """İlk kare üzerinde kullanıcıdan 2 ROI seçimi al."""
        ok, first_frame = self._read_frame()
        if not ok:
            raise RuntimeError("Kameradan ilk kare okunamadı.")

        selector = ROISelector(first_frame)
        polygons = selector.run()

        self.regions = [
            Region(index=i, polygon=poly)
            for i, poly in enumerate(polygons)
        ]

    # ── Tespit işleme ─────────────────────────
    def _process_detections(self, frame: np.ndarray, results) -> None:
        """
        Her kişi için:
        1. Ayak noktasını hesapla
        2. Hangi bölgede olduğunu bul
        3. ID'yi o bölgeye kaydet (seen_ids)
        4. Kutuyu ve ID'yi çiz
        """
        for result in results:
            if result.boxes.id is None:
                continue  # ByteTrack henüz ID atayamadıysa atla

            for box, track_id in zip(result.boxes, result.boxes.id):
                tid  = int(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                foot = (int((x1 + x2) / 2), y2)  # Kutunun alt-orta noktası

                matched: Optional[Region] = None
                for region in self.regions:
                    if region.contains(foot):
                        region.register(tid)
                        matched = region
                        break

                color = matched.color if matched else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ── HUD (bilgi paneli) ────────────────────
    def _draw_hud(self, frame: np.ndarray) -> None:
        """Bölge poligonlarını ve kişi sayımlarını ekranda göster."""
        for region in self.regions:
            cv2.polylines(frame, [region.polygon], True, region.color, 2)

        panel_h = 20 + len(self.regions) * 30
        cv2.rectangle(frame, (0, 0), (240, panel_h), (0, 0, 0), -1)
        for i, region in enumerate(self.regions):
            cv2.putText(frame,
                        f"{region.label}: {region.count} kisi",
                        (10, 20 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, region.color, 2)

    # ── Özet yazdır ───────────────────────────
    def _print_summary(self) -> None:
        print("\n── Özet ──────────────────────")
        for region in self.regions:
            print(f"  {region.label}: {region.count} benzersiz kişi")
        print("──────────────────────────────")

    # ── Ana döngü ─────────────────────────────
    def run(self) -> None:
        win = "Takip Ekrani"
        print("Takip başladı. Çıkmak için 'Q' veya pencereyi kapatın.")

        while self.cap.isOpened():
            ok, frame = self._read_frame()
            if not ok:
                print("Kamera bağlantısı kesildi.")
                break

            # YOLOv8 + ByteTrack
            results = self.model.track(
                frame,
                conf=self.cfg.confidence,
                classes=[self.cfg.person_class],
                tracker=self.cfg.tracker,
                persist=True,    # ID sürekliliği için zorunlu
                verbose=False,
            )

            self._process_detections(frame, results)
            self._draw_hud(frame)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            pencere_kapandi = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1

            if key == ord('q') or pencere_kapandi:
                self._print_summary()
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Giriş Noktası
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg     = Config()   # rtsp_url ve diğer ayarları burada değiştirin
    counter = PeopleCounter(cfg)
    counter.setup_regions()
    counter.run()
