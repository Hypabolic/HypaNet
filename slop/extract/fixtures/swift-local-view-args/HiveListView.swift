import SwiftUI

struct HiveListView: View {
    @State private var loadError: String?
    var body: some View {
        VStack {
            if let loadError {
                HiveNoticeView(title: "Hives did not load", message: loadError)
            }
            HiveNoticeView(title: "No hives yet", message: "Add your first hive to start logging inspections.")
            PairingCodeView(text: "hive-7731", caption: "Scan to pair a scale")
            RemoteNoticeView(title: "Swarm season", message: loadError ?? "")
        }
    }
}

struct HiveNoticeView: View {
    let title: String
    let message: String
    var body: some View {
        VStack {
            Text(title).font(.headline)
            Text(message.uppercased())
        }
    }
}

struct PairingCodeView: View {
    let text: String
    let caption: String
    var body: some View {
        VStack {
            Image(uiImage: QRMaker.image(for: text))
            Text(caption)
        }
    }
}
