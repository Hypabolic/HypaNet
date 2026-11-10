import SwiftUI

struct PantryShelfView: View {
    var body: some View {
        VStack {
            Text("Shelf jars")
            #if DEBUG
            Button("Reset shelf cache") {}
            #endif
            Text("Label the lids before storing")
            Spacer()
        }
    }
}
