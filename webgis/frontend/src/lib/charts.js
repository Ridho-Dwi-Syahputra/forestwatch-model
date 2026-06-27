// Setup Chart.js (registrasi sekali) + builder data/opsi chart yang dipakai beberapa halaman.
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend as ChartLegend,
  LinearScale,
  Tooltip,
} from "chart.js";

import { PROVINCE_CHART_COLORS } from "./constants";
import { formatNumber } from "./format";

// Registrasi global -- dijalankan sekali saat modul ini pertama di-import.
ChartJS.register(ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, ChartLegend);

export const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (context) => `${context.label}: ${formatNumber(context.raw, 1)} ha`,
      },
    },
  },
};

export const barOptions = {
  ...chartOptions,
  indexAxis: "y",
  scales: {
    x: { grid: { color: "#e3ece6" }, ticks: { color: "#53645b" } },
    y: { grid: { display: false }, ticks: { color: "#314239" } },
  },
};

export function buildDoughnutData(transitionRows) {
  return {
    labels: transitionRows.map((row) => row.label),
    datasets: [
      {
        data: transitionRows.map((row) => row.value),
        backgroundColor: transitionRows.map((row) => row.chartColor),
        borderColor: "#ffffff",
        borderWidth: 3,
      },
    ],
  };
}

export function buildProvinceBarData(provinceRows) {
  return {
    labels: provinceRows.map((row) => row.province),
    datasets: [
      {
        label: "ha",
        data: provinceRows.map((row) => row.deforestation_ha),
        backgroundColor: provinceRows.map(
          (_, index) => PROVINCE_CHART_COLORS[index % PROVINCE_CHART_COLORS.length]
        ),
        borderRadius: 6,
        barThickness: 18,
      },
    ],
  };
}
