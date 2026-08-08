import next from "eslint-config-next";

const config = [
  ...next,
  {
    ignores: ["node_modules/**", ".next/**", "out/**", "public/sw.js", "next-env.d.ts"],
  },
];

export default config;
