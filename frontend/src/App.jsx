import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';

function App() {
  const [formData, setFormData] = useState({
    Gender: '',
    Married: '',
    Dependents: '',
    Education: '',
    Self_Employed: '',
    ApplicantIncome: '',
    CoapplicantIncome: '',
    LoanAmount: '',
    Loan_Amount_Term: '',
    Credit_History: '',
    Property_Area: '',
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setPrediction(data.prediction ?? data.error);
    } catch (error) {
      setPrediction('Server Error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-200 via-white to-pink-100 flex items-center justify-center px-4 py-12 font-sans">
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-4xl bg-white/80 backdrop-blur-lg shadow-2xl rounded-3xl p-10"
      >
        <motion.h2
          className="text-4xl font-extrabold text-center text-indigo-700 mb-10"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
        >
          🏦 Premium Loan Approval Predictor
        </motion.h2>

        <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            ['Gender', ['Male', 'Female']],
            ['Married', ['Yes', 'No']],
            ['Dependents', ['0', '1', '2', '3+']],
            ['Education', ['Graduate', 'Not Graduate']],
            ['Self_Employed', ['Yes', 'No']],
            ['Property_Area', ['Urban', 'Semiurban', 'Rural']],
            ['Credit_History', ['1', '0']],
          ].map(([label, options]) => (
            <div key={label}>
              <label className="block mb-1 text-sm font-medium text-gray-700">{label}</label>
              <select
                name={label}
                value={formData[label]}
                onChange={handleChange}
                required
                className="w-full p-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400"
              >
                <option value="">Select {label}</option>
                {options.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>
          ))}

          {[
            ['ApplicantIncome', 'Income of applicant'],
            ['CoapplicantIncome', 'Income of co-applicant'],
            ['LoanAmount', 'Requested loan amount'],
            ['Loan_Amount_Term', 'Loan term (in days)'],
          ].map(([name, placeholder]) => (
            <div key={name}>
              <label className="block mb-1 text-sm font-medium text-gray-700">{placeholder}</label>
              <input
                type="number"
                name={name}
                value={formData[name]}
                onChange={handleChange}
                placeholder={placeholder}
                className="w-full p-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400"
                required
              />
            </div>
          ))}

          <div className="md:col-span-2 text-center mt-4">
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              type="submit"
              className="bg-indigo-600 text-white font-bold py-3 px-10 rounded-xl shadow hover:bg-indigo-700 transition duration-300"
            >
              {loading ? 'Predicting...' : 'Predict'}
            </motion.button>
          </div>
        </form>

        {prediction !== null && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className={`mt-8 flex flex-col items-center ${
              prediction === 1 || prediction === '1'
                ? 'text-green-600'
                : 'text-red-600'
            }`}
          >
            {prediction === '1' || prediction === 1 ? (
              <>
                <FaCheckCircle className="text-4xl mb-2" />
                <span className="text-xl font-semibold">Loan Approved ✅</span>
              </>
            ) : prediction === '0' || prediction === 0 ? (
              <>
                <FaTimesCircle className="text-4xl mb-2" />
                <span className="text-xl font-semibold">Loan Rejected ❌</span>
              </>
            ) : (
              <span className="text-lg font-semibold text-yellow-600">{prediction}</span>
            )}
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}

export default App;
