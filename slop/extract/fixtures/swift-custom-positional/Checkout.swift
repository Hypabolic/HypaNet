import SwiftUI

struct Checkout: View {
    let model: CheckoutModel

    var body: some View {
        VStack {
            SummaryCard(copy: ReceiptCopy(model.order, currency: "GBP"))
            TagPill("Gift wrapped", systemImage: "gift")
            FacetChip(String(localized: "Express"), systemImage: "bolt") { model.toggleExpress() }
            FieldHeader("Delivery address")
            point("shippingbox", "Parcels leave the depot before noon.")
            point("leaf", "Packaging is recycled card.")
            PrimaryButton("Pay now", systemImage: "creditcard") { model.pay() }
            SecondaryButton("Back to basket") { model.back() }
        }
    }

    private func point(_ systemImage: String, _ text: String) -> some View {
        Label {
            Text(text)
                .font(.callout)
        } icon: {
            Image(systemName: systemImage)
        }
    }
}
