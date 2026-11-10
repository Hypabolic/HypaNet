import SwiftUI

struct InvoiceListView: View {
    @State private var invoices: [Invoice] = []
    @State private var confirmDelete = false
    let exportCount: Int

    var body: some View {
        NavigationStack {
            Group {
                if invoices.isEmpty {
                    VStack(spacing: 12) {
                        Text("No invoices")
                            .font(.title2)
                        Text("There are no invoices yet.")
                            .foregroundStyle(.secondary)
                        Button("Create invoice") { create() }
                    }
                } else {
                    List(invoices) { invoice in
                        InvoiceRow(invoice: invoice)
                    }
                }
            }
            .navigationTitle("Invoices")
            .toolbar {
                Button("Delete all") { confirmDelete = true }
            }
            .alert("Delete all invoices?", isPresented: $confirmDelete) {
                Button("Delete all invoices", role: .destructive) { deleteAll() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This also deletes \(exportCount) saved exports.")
            }
        }
    }
}
