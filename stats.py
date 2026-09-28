from pathlib import Path
from collections import defaultdict
import json
from time import ctime

current = Path(__file__).parent

EXTENSIONS = {".js", ".css", ".html", ".htm", ".py", ".txt"}


def collect_file_stats(base: Path):
    results = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            # Skip files we can't stat
            continue

        lines = 0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for _ in f:
                    lines += 1
        except OSError:
            # If unreadable as text, still include size/mtime
            pass

        results.append(
            {
                "path": str(path.relative_to(base)),
                "ext": path.suffix.lower(),
                "size_bytes": size,
                "modified": mtime,
                "modified_readable": ctime(mtime),
                "lines": lines,
            }
        )
    return results


def summarize(stats):
    summary = {
        "total_files": len(stats),
        "total_bytes": sum(s["size_bytes"] for s in stats),
        "total_lines": sum(s["lines"] for s in stats),
        "by_extension": {},
    }
    by_ext = defaultdict(lambda: {"files": 0, "bytes": 0, "lines": 0})
    for s in stats:
        e = s["ext"]
        by_ext[e]["files"] += 1
        by_ext[e]["bytes"] += s["size_bytes"]
        by_ext[e]["lines"] += s["lines"]
    summary["by_extension"] = dict(by_ext)
    return summary


def main():
    files = collect_file_stats(current)
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine

    try:
        # Prefer a non-native style so custom backgrounds work without warnings
        from PySide6.QtQuickControls2 import QQuickStyle

        QQuickStyle.setStyle("Basic")
    except Exception:
        pass

    app = QGuiApplication([])
    summary = summarize(files)
    by_ext = [
        {
            "ext": ext,
            "files": data["files"],
            "bytes": data["bytes"],
            "lines": data["lines"],
        }
        for ext, data in sorted(
            summary["by_extension"].items(), key=lambda x: (-x[1]["files"], x[0])
        )
    ]

    qml = """
    import QtQuick 2.15
    import QtQuick.Controls 2.15
    import QtQuick.Layouts 1.15

    ApplicationWindow {
        id: win
        width: 980
        height: 640
        visible: true
        title: "Project Stats"
        color: "#0f1115"

        header: ToolBar {
            background: Rectangle { color: "#1b1f2a" }
            RowLayout {
                anchors.fill: parent
                spacing: 16
                Label {
                    text: "Project Stats"
                    font.pixelSize: 18
                    color: "white"
                    Layout.margins: 12
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: summary ? (summary.total_files + " files") : ""
                    color: "#c0c4d0"
                    Layout.margins: 12
                }
                Label {
                    text: summary ? (Number(summary.total_lines).toLocaleString(Qt.locale()) + " lines") : ""
                    color: "#c0c4d0"
                    Layout.margins: 12
                }
                Label {
                    text: summary ? humanBytes(summary.total_bytes) : ""
                    color: "#c0c4d0"
                    Layout.margins: 12
                }
            }
        }

        function humanBytes(n) {
            var u=["B","KB","MB","GB","TB"]
            var i=0
            var x=n
            while (x>=1024 && i<u.length-1) { x/=1024; i++ }
            var d=(x<10 && i>0)?1:0
            return x.toFixed(d) + " " + u[i]
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 12
            spacing: 12

            Flickable {
                Layout.fillWidth: true
                Layout.preferredHeight: 80
                contentWidth: extRow.implicitWidth
                contentHeight: extRow.implicitHeight
                clip: true
                ScrollBar.horizontal: ScrollBar { }

                Row {
                    id: extRow
                    spacing: 8
                    Repeater {
                        model: byExtension
                        delegate: Rectangle {
                            radius: 8
                            color: "#202638"
                            border.color: "#3a425a"
                            border.width: 1
                            implicitHeight: 40
                            implicitWidth: chipText.implicitWidth + 24

                            Text {
                                id: chipText
                                anchors.centerIn: parent
                                color: "#d7dbea"
                                font.pixelSize: 14
                                text: modelData.ext + " • " + modelData.files + " • " + humanBytes(modelData.bytes)
                            }

                            ToolTip.visible: ma.containsMouse
                            ToolTip.text: Number(modelData.lines).toLocaleString(Qt.locale()) + " lines"
                            MouseArea { id: ma; anchors.fill: parent; hoverEnabled: true }
                        }
                    }
                }
            }

            Frame {
                Layout.fillWidth: true
                Layout.fillHeight: true
                background: Rectangle { radius: 10; color: "#141824"; border.color: "#2a3145"; border.width: 1 }
                // Wrap the ListView to provide margins without using unsupported padding
                Item {
                    anchors.fill: parent
                    anchors.margins: 12

                    ListView {
                        id: list
                        anchors.fill: parent
                        clip: true
                        model: files
                        spacing: 8
                        ScrollBar.vertical: ScrollBar { }

                        delegate: Rectangle {
                            width: list.width
                            height: 76
                            radius: 10
                            color: "#1a2031"
                            border.color: "#2e3650"
                            border.width: 1

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 4

                                Text {
                                    text: modelData.path
                                    color: "#e6e9f2"
                                    font.pixelSize: 15
                                    elide: Text.ElideMiddle
                                }
                                RowLayout {
                                    spacing: 16
                                    Text { text: modelData.ext; color: "#9aa3ba"; font.pixelSize: 13 }
                                    Text { text: Number(modelData.lines).toLocaleString(Qt.locale()) + " lines"; color: "#9aa3ba"; font.pixelSize: 13 }
                                    Text { text: humanBytes(modelData.size_bytes); color: "#9aa3ba"; font.pixelSize: 13 }
                                    Text { text: modelData.modified_readable; color: "#9aa3ba"; font.pixelSize: 13 }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """

    engine = QQmlApplicationEngine()
    ctx = engine.rootContext()
    ctx.setContextProperty("files", files)
    ctx.setContextProperty("summary", summary)
    ctx.setContextProperty("byExtension", by_ext)
    engine.loadData(qml.encode("utf-8"))
    if not engine.rootObjects():
        raise SystemExit("Failed to load UI")
    app.exec()


if __name__ == "__main__":
    main()
