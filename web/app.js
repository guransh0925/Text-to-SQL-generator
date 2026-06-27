const form = document.querySelector("#generatorForm");
const schemaInput = document.querySelector("#schema");
const questionInput = document.querySelector("#question");
const result = document.querySelector("#result code");
const statusText = document.querySelector("#status");
const generateButton = document.querySelector("#generateButton");
const sampleButton = document.querySelector("#sampleButton");
const copyButton = document.querySelector("#copyButton");

const samples = [
  {
    schema: `CREATE TABLE customers (
  id INTEGER PRIMARY KEY,
  name TEXT,
  city TEXT,
  signup_date TEXT
);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  order_date TEXT,
  total REAL,
  status TEXT
);`,
    question: "Show completed orders over 1000.",
  },
  {
    schema: `Table: employees
Columns: employee_id, employee_name, department, salary`,
    question: "Retrieve the names of all employees earning more than $50,000.",
  },
];

async function loadDefaults() {
  try {
    const response = await fetch("/api/defaults");
    const data = await response.json();
    schemaInput.value = data.schema || samples[0].schema;
    questionInput.value = samples[0].question;
  } catch {
    schemaInput.value = samples[0].schema;
    questionInput.value = samples[0].question;
  }
}

function setBusy(isBusy) {
  generateButton.disabled = isBusy;
  generateButton.textContent = isBusy ? "Generating..." : "Generate SQL";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  statusText.textContent = "Sending this to your model...";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        schema: schemaInput.value,
        question: questionInput.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }
    result.textContent = data.sql || "-- No SQL returned.";
    statusText.textContent = "Done. Review the query before running it on a real database.";
  } catch (error) {
    result.textContent = "-- Generation failed.";
    statusText.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

sampleButton.addEventListener("click", () => {
  const currentIndex = samples.findIndex((sample) => sample.question === questionInput.value);
  const sample = samples[(currentIndex + 1) % samples.length];
  schemaInput.value = sample.schema;
  questionInput.value = sample.question;
  result.textContent = "SELECT ...";
  statusText.textContent = "Sample loaded.";
});

copyButton.addEventListener("click", async () => {
  const sql = result.textContent.trim();
  if (!sql || sql === "SELECT ...") {
    statusText.textContent = "Generate SQL first, then copy it.";
    return;
  }

  try {
    await navigator.clipboard.writeText(sql);
    statusText.textContent = "Copied.";
  } catch {
    statusText.textContent = "Copy is not available in this browser.";
  }
});

loadDefaults();
