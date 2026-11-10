import { useTranslation } from "react-i18next";

export function Greeting({ user, items, order }: Props) {
  const { t } = useTranslation();
  return (
    <div>
      <h2>Morning, {user.firstName}</h2>
      <p>{`You have ${items.length} parcels waiting at the depot`}</p>
      <p>{t("account.settingsTitle")}</p>
      <button>{t("Save changes")}</button>
      <span>Last collected {formatDate(order.createdAt)}</span>
      <span>{user?.profile.displayName}</span>
      <span>{items.length} of {order.capacity} slots used</span>
    </div>
  );
}
