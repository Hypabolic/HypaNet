import styles from "./Banner.module.css";

export function Banner() {
  return (
    <div className={styles.banner}>
      <p className={styles.eyebrowLine}>Members only</p>
      <h2 className={styles.title}>Early entry to the autumn plant sale</h2>
    </div>
  );
}
