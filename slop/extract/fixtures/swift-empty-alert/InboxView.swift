import SwiftUI

struct InboxView: View {
    @State private var confirmClear = false
    let letters: [Letter]

    var body: some View {
        Group {
            if letters.isEmpty {
                ContentUnavailableView("No letters", systemImage: "envelope",
                    description: Text("Letters from your pen pals land here."))
            } else {
                List(letters) { letter in
                    Text(letter.subject)
                }
            }
        }
        .alert("Clear the inbox?", isPresented: $confirmClear) {
            Button("Clear", role: .destructive) { clear() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Archived letters are not affected.")
        }
        .confirmationDialog("Sort letters", isPresented: $confirmClear) {
            Button("Newest first") {}
            Button("By sender") {}
        }
    }
}
