// Helper format angka & label (locale Indonesia).
import { TRANSITION_META } from "./constants";

export function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return new Intl.NumberFormat("id-ID", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number(value));
}

export function formatPercent(value) {
  if (value === null || value === undefined) return "-";
  return `${formatNumber(Number(value) * 100, 0)}%`;
}

export function transitionLabel(type) {
  return TRANSITION_META[type]?.longLabel || type?.replaceAll("_", " ") || "-";
}
