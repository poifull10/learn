import QtQuick 2.5
import QtQuick.Window 2.2

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onClicked: {
            Qt.quit();
        }
        onEntered: color = "lightblue"
        onExited: color = "white"
    }

    Text {
        text: hoge.getTextFromCpp()
        anchors.centerIn: parent
    }
}