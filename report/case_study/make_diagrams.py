"""Build the case-study diagrams as draw.io files and render each to PNG.

Every diagram is placed by hand so the layout is deliberate rather than whatever
an auto-layout pass decides. Rendering goes through the official draw.io viewer
in headless Chrome, so the PNG in the report is exactly what draw.io shows when
the .drawio file is opened.

    python report/case_study/make_diagrams.py
"""
import base64
import json
import subprocess
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "diagrams"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
VIEWER = "https://viewer.diagrams.net/js/viewer-static.min.js"

FILLS = {
    "input":    ("#F5F5F5", "#666666"),
    "step":     ("#DAE8FC", "#6C8EBF"),
    "model":    ("#E1D5E7", "#9673A6"),
    "extra":    ("#FFE6CC", "#D79B00"),
    "output":   ("#D5E8D4", "#82B366"),
    "bad":      ("#F8CECC", "#B85450"),
    "decision": ("#FFF2CC", "#D6B656"),
}
FONT = "fontFamily=Helvetica;fontColor=#1A1A1A;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;"
        "strokeWidth=2;strokeColor=#555555;fontSize=14;" + FONT +
        "labelBackgroundColor=#FFFFFF;")


class Diagram:
    def __init__(self, name):
        self.name = name
        self.cells = []
        self.next_id = 2
        self.right = self.bottom = 0

    def _id(self):
        self.next_id += 1
        return f"n{self.next_id}"

    def box(self, label, x, y, w=170, h=60, kind="step", shape="rounded", size=16, bold=False):
        fill, stroke = FILLS[kind]
        style = {
            "rounded": "rounded=1;arcSize=14;",
            "diamond": "rhombus;",
            "ellipse": "ellipse;",
            "cylinder": "shape=cylinder3;boundedLbl=1;size=10;",
        }[shape]
        style += (f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                  f"strokeWidth=2;fontSize={size};{FONT}" + ("fontStyle=1;" if bold else ""))
        return self._vertex(label, x, y, w, h, style)

    def group(self, label, x, y, w, h):
        style = ("rounded=1;arcSize=4;html=1;dashed=1;fillColor=none;strokeColor=#999999;"
                 f"strokeWidth=2;verticalAlign=top;align=left;spacingLeft=12;spacingTop=4;"
                 f"fontSize=15;fontStyle=1;fontColor=#555555;fontFamily=Helvetica;")
        return self._vertex(label, x, y, w, h, style, back=True)

    def text(self, label, x, y, w=160, h=30, size=14, bold=False):
        color = "#1A1A1A" if bold else "#555555"
        style = (f"text;html=1;align=center;verticalAlign=middle;fontSize={size};fontColor={color};"
                 "fontFamily=Helvetica;" + ("fontStyle=1;" if bold else ""))
        return self._vertex(label, x, y, w, h, style)

    def image(self, name, x, y, w, h):
        # draw.io takes data URIs without ";base64" because ";" separates style keys.
        data = base64.b64encode((HERE / "assets" / f"{name}.jpg").read_bytes()).decode()
        style = ("shape=image;html=1;imageAspect=0;imageBorder=#999999;strokeWidth=2;"
                 f"image=data:image/jpeg,{data};")
        return self._vertex("", x, y, w, h, style)

    def _vertex(self, label, x, y, w, h, style, back=False):
        cell_id = self._id()
        xml = (f'<mxCell id="{cell_id}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
               f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        # Groups go first so boxes draw on top of their dashed outline.
        self.cells.insert(0, xml) if back else self.cells.append(xml)
        self.right = max(self.right, x + w)
        self.bottom = max(self.bottom, y + h)
        return cell_id

    def arrow(self, src, dst, label="", exit=None, entry=None, dashed=False):
        style = EDGE + ("dashed=1;" if dashed else "")
        for side, point in (("exit", exit), ("entry", entry)):
            if point:
                style += f"{side}X={point[0]};{side}Y={point[1]};{side}Dx=0;{side}Dy=0;"
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{escape(label)}" style="{style}" edge="1" '
            f'parent="1" source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>')

    def xml(self):
        return ('<mxGraphModel grid="0" page="0" background="#FFFFFF"><root>'
                '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) +
                "</root></mxGraphModel>")

    def save(self):
        model = self.xml()
        (OUT / f"{self.name}.drawio").write_text(
            f'<mxfile host="drawio"><diagram name="{self.name}">{model}</diagram></mxfile>',
            encoding="utf-8")
        self._render(model)

    def _render(self, model):
        pad = 20
        width, height = self.right + 2 * pad, self.bottom + 2 * pad
        config = {"xml": model, "nav": False, "resize": False, "border": pad,
                  "highlight": "none", "lightbox": False}
        page = OUT / f"{self.name}.html"
        page.write_text(
            "<!doctype html><html><body style='margin:0;background:#fff'>"
            f"<div class='mxgraph' data-mxgraph='{escape(json.dumps(config), quote=True)}'></div>"
            f"<script src='{VIEWER}'></script></body></html>", encoding="utf-8")
        png = OUT / f"{self.name}.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--window-size={width},{height}",
                        "--virtual-time-budget=20000", f"--screenshot={png}", page.as_uri()],
                       check=True, capture_output=True)
        page.unlink()
        print(f"{png.name}: {width}x{height}")


def how_it_works():
    """The whole system in plain words, with real images from one held-out clip."""
    d = Diagram("fig0_how_it_works")
    frame_style = ("rounded=1;arcSize=6;html=1;fillColor=#FAFAFA;strokeColor=#BBBBBB;"
                   "strokeWidth=1.5;verticalAlign=top;fontSize=18;fontStyle=1;spacingTop=8;" + FONT)

    d.group("Prepare the input", 0, 0, 1120, 560)
    step1 = d._vertex("1  Film the sign", 40, 45, 340, 235, frame_style)
    d.image("frame", 60, 90, 300, 169)
    step2 = d._vertex("2  Keep 16 frames", 440, 45, 540, 235, frame_style)
    d.image("filmstrip", 460, 90, 500, 172)
    d.arrow(step1, step2)

    step3 = d._vertex("3  Cut out hands and face", 40, 320, 420, 205, frame_style)
    for i, (name, label) in enumerate((("crop_left", "Left hand"), ("crop_right", "Right hand"),
                                       ("crop_face", "Face"))):
        d.image(name, 70 + i * 125, 365, 100, 100)
        d.text(label, 60 + i * 125, 470, 120, 26, size=14)
    step4 = d._vertex("4  Mark hand joints", 580, 320, 330, 225, frame_style)
    d.image("joints", 650, 362, 190, 170)
    d.arrow(step2, step3, exit=(0.3, 1), entry=(0.5, 0))
    d.arrow(step2, step4, exit=(0.6, 1), entry=(0.5, 0))

    d.group("Recognise the sign", 0, 610, 1120, 280)
    spatial = d.box("5  Study each frame", 135, 720, 230, 80, kind="model", bold=True, size=17)
    temporal = d.box("6  Follow the motion", 425, 720, 230, 80, kind="model", bold=True, size=17)
    sure = d.box("Sure?", 705, 705, 130, 110, kind="decision", shape="diamond", size=17)
    word = d.box("“loud”", 885, 665, 190, 70, kind="output", bold=True, size=22)
    again = d.box("Please repeat", 885, 785, 190, 70, kind="bad", size=17)
    d.arrow(step3, spatial, exit=(0.5, 1), entry=(0.35, 0))
    d.arrow(step4, spatial, exit=(0.5, 1), entry=(0.75, 0))
    d.arrow(spatial, temporal)
    d.arrow(temporal, sure)
    d.arrow(sure, word, "yes")
    d.arrow(sure, again, "no")
    d.save()


def system_overview():
    d = Diagram("fig1_system_overview")
    video = d.box("Sign video", 0, 110, 140, kind="input")
    detect = d.box("Find hands and face", 190, 110, 170)
    crops = d.box("Hand and face crops", 410, 40, 180, kind="extra")
    joints = d.box("Hand landmarks", 410, 180, 180, kind="extra")
    model = d.box("ISL-ViT-Tiny", 640, 95, 170, 90, kind="model", bold=True, size=18)
    gate = d.box("Confident?", 860, 85, 150, 110, kind="decision", shape="diamond")
    word = d.box("Show word", 1070, 30, 150, kind="output")
    repeat = d.box("Ask to repeat", 1070, 190, 150, kind="bad")
    d.arrow(video, detect)
    d.arrow(detect, crops)
    d.arrow(detect, joints)
    d.arrow(crops, model)
    d.arrow(joints, model)
    d.arrow(model, gate)
    d.arrow(gate, word, "yes")
    d.arrow(gate, repeat, "no")
    d.save()


def model_architecture():
    d = Diagram("fig2_model_architecture")
    d.group("Spatial stage", 0, 0, 600, 330)
    left = d.box("Left hand", 30, 45, 160, kind="input")
    right = d.box("Right hand", 220, 45, 160, kind="input")
    face = d.box("Face", 410, 45, 160, kind="input")
    patches = d.box("Patch embedding", 150, 150, 300)
    spatial = d.box("Spatial encoder ×4", 150, 245, 300, kind="model", bold=True)
    for stream in (left, right, face):
        d.arrow(stream, patches)
    d.arrow(patches, spatial)

    # Centred on the crop-token row so the four arrows meet it without crossing a group.
    d.group("Added per crop", 760, 230, 250, 360)
    extras = [d.box(label, 800, 270 + i * 76, 170, 56, kind="extra")
              for i, label in enumerate(("Box position", "Detection flag", "Handshape", "Body pose"))]
    enrich = d.box("Crop tokens", 150, 400, 300, kind="step")
    d.arrow(spatial, enrich)
    for extra in extras:
        d.arrow(extra, enrich, exit=(0, 0.5), entry=(1, 0.5))

    d.group("Temporal stage", 0, 630, 600, 330)
    fusion = d.box("Tag stream and time", 150, 675, 300)
    temporal = d.box("Temporal encoder ×4", 150, 770, 300, kind="model", bold=True)
    head = d.box("Classify from class token", 150, 865, 300)
    words = d.box("50 words", 800, 865, 170, kind="output", bold=True)
    d.arrow(enrich, fusion)
    d.arrow(fusion, temporal)
    d.arrow(temporal, head)
    d.arrow(head, words)
    d.save()


def evaluation_split():
    d = Diagram("fig3_evaluation_split")
    clips = d.box("INCLUDE clips", 330, 0, 200, kind="input")
    numbers = d.box("Camera clip numbers", 330, 100, 200)
    sessions = d.box("13 sessions", 330, 200, 200, kind="model", bold=True)
    random = d.box("Random split", 90, 320, 200, kind="bad")
    session = d.box("Session split", 570, 320, 200, kind="output")
    shared = d.box("Same sessions both sides", 90, 420, 200, 70, kind="bad")
    unseen = d.box("Test sessions unseen", 570, 420, 200, 70, kind="output")
    high = d.box("98.5 %", 90, 530, 200, kind="bad", bold=True, size=20)
    honest = d.box("75.6 %", 570, 530, 200, kind="output", bold=True, size=20)
    d.text("leakage", 330, 535, 200, 50, size=18)
    for src, dst in ((clips, numbers), (numbers, sessions), (random, shared),
                     (session, unseen), (shared, high), (unseen, honest)):
        d.arrow(src, dst)
    d.arrow(sessions, random, exit=(0, 0.5), entry=(0.5, 0))
    d.arrow(sessions, session, exit=(1, 0.5), entry=(0.5, 0))
    d.save()


def training_pipeline():
    d = Diagram("fig4_training_pipeline")
    d.group("Pretraining", 0, 0, 700, 130)
    isign = d.box("iSign videos", 30, 45, 170, kind="input")
    masked = d.box("Masked pretraining", 260, 45, 190)
    weights = d.box("Encoder weights", 510, 45, 160, kind="model")
    d.arrow(isign, masked)
    d.arrow(masked, weights)

    d.group("Training", 0, 180, 1180, 130)
    include = d.box("INCLUDE clips", 30, 225, 170, kind="input")
    train = d.box("Train 262 words", 260, 225, 190)
    keep = d.box("Keep 50 words", 510, 225, 160)
    qat = d.box("4-bit tuning", 730, 225, 170, kind="extra")
    ship = d.box("2 MB model", 960, 225, 190, kind="output", bold=True)
    d.arrow(weights, train, exit=(0.5, 1), entry=(0.5, 0))
    for src, dst in ((include, train), (train, keep), (keep, qat), (qat, ship)):
        d.arrow(src, dst)
    d.save()


def quantisation():
    d = Diagram("fig5_quantisation")
    full = d.box("Full model 14.9 MB", 0, 60, 180, kind="input")
    groups = d.box("Groups of 128", 230, 60, 170)
    rounding = d.box("Round to 4 bits", 450, 60, 170, kind="extra")
    tune = d.box("Train with rounding", 670, 60, 180, kind="model")
    pack = d.box("Compact file", 900, 60, 160)
    small = d.box("1.97 MB model", 1110, 60, 170, kind="output", bold=True)
    for src, dst in ((full, groups), (groups, rounding), (rounding, tune), (tune, pack), (pack, small)):
        d.arrow(src, dst)
    d.arrow(tune, rounding, "300 epochs", exit=(0.5, 0), entry=(0.5, 0), dashed=True)
    d.save()


def landmark_features():
    d = Diagram("fig6_landmark_features")
    joints = d.box("21 hand joints", 0, 40, 170, kind="input")
    pose = d.box("Body pose", 0, 180, 170, kind="input")
    shape = d.box("Handshape", 230, 0, 170, kind="extra")
    place = d.box("Sign location", 230, 100, 170, kind="extra")
    body = d.box("Body frame", 230, 200, 170, kind="extra")
    mlp = d.box("Small network", 460, 90, 170)
    token = d.box("Add to crop token", 690, 90, 190, kind="model", bold=True)
    d.text("wrist centred, size scaled", 190, 60, 250, 30, size=13)
    d.arrow(joints, shape)
    d.arrow(joints, place, entry=(0, 0.3))
    d.arrow(pose, place, exit=(1, 0.2), entry=(0, 0.75))
    d.arrow(pose, body)
    for feature in (shape, place, body):
        d.arrow(feature, mlp)
    d.arrow(mlp, token)
    d.save()


def research_gap():
    d = Diagram("fig7_research_gap")
    ellipse = ("ellipse;whiteSpace=wrap;html=1;strokeWidth=2;fontSize=17;fontStyle=1;"
               "opacity=45;textOpacity=100;" + FONT)
    circles = (("RGB transformer", 0, 0, "#6C8EBF", "top"),
               ("Tiny model", 220, 0, "#D79B00", "top"),
               ("Unseen signers", 110, 190, "#82B366", "bottom"))
    for label, x, y, color, anchor in circles:
        d._vertex(label, x, y, 340, 320,
                  ellipse + f"fillColor={color};strokeColor={color};verticalAlign={anchor};"
                  f"spacingTop=45;spacingBottom=45;")
    d.box("This work", 205, 215, 150, 60, kind="model", bold=True, size=18)
    d.save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (how_it_works, system_overview, model_architecture, evaluation_split, training_pipeline,
                  quantisation, landmark_features, research_gap):
        build()
