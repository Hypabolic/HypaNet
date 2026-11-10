import SwiftUI

struct RouteList: View {
    let routes: [Route]

    var body: some View {
        List {
            ForEach(routes) { route in
                NavigationLink(value: route) {
                    VStack(alignment: .leading) {
                        Text(route.name)
                            .font(.headline)
                        Text("\(route.distance) km · \(route.climb) m climb")
                            .font(.caption)
                    }
                }
                .badge(route.unread)
                .swipeActions {
                    Button("Archive") { archive(route) }
                }
            }
        }
        .navigationTitle("Saved routes")
    }
}
