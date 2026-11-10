import SwiftUI

struct DeviceRow: View {
    let device: Device

    var body: some View {
        HStack {
            Circle()
                .fill(device.isOnline ? Color.green : Color.gray)
                .frame(width: 8, height: 8)
            Text(device.isOnline ? "Connected" : "Not connected")
            Spacer()
            Text("Beta")
                .font(.caption2)
                .padding(.horizontal, 6)
                .background(Capsule().fill(.blue.opacity(0.2)))
            Text(device.serial)
                .monospaced()
                .lineLimit(1)
        }
    }
}
