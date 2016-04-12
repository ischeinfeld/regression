package regression;

import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;

public class PolynomialRegressionModel extends RegressionModel {

	private boolean degreeSet;
	
	/** The coefficients of the polynomial */
	private double[] coefficients;

	/**
	 * Construct a new PolynomialRegressionModel with the supplied data set
	 * @param x The x data points
	 * @param y The y data points
	 */
	public PolynomialRegressionModel(double[] x, double[] y) {
		super(x, y);
		degreeSet = false;
	}

	/**
	 * set the degree of the fitted polynomial
	 */
	public void setDegree(int degree) {

		// throws exception if degree is not a positive integer
		if (degree < 0) {
			throw new IllegalArgumentException("Degree must be positive");
		}

		coefficients = new double[degree + 1];
		degreeSet = true;
		computed = false; // model must be computed again with new degree
	}
	
	/**
	 * Get the coefficients of the fitted straight line
	 * @return An array of coefficients {intercept, gradient}
	 * @see RegressionModel#getCoefficients()
	 */
	@Override
	public double[] getCoefficients() {
		if (!degreeSet) {
			throw new IllegalStateException("Degree of polynomial has not been set and model has not yet computed");
		}
		else if (degreeSet && !computed) {
			throw new IllegalStateException("Model has not yet computed");
		}
		
		return coefficients;
	}

	/**
	 * Compute the coefficients of a straight line the best fits the data set
	 * @see RegressionModel#compute()
	 */
	@Override
	public void compute() {

		if (!degreeSet) {
			throw new IllegalStateException("Degree of polynomial has not been set");
		}
		
		// throws exception if regression can not be performed
		if (xValues.length < 2 | yValues.length < 2) {
			throw new IllegalArgumentException("Must have more than two values");
		}
		
		// fits curve to data
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		
		for(int i=0; i<this.xValues.length; i++) {
			obs.add(this.xValues[i], this.yValues[i]);
		}
		
		final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(coefficients.length-1);

		coefficients = fitter.fit(obs.toList());
		
		// set the computed flag to true after we have calculated the coefficients
		computed = true;
	}
	
	/**
	 * Compute the coefficients of a straight line the best fits the data set
	 * @see RegressionModel#compute()
	 */
	public void computeDevel(double lambda) {

		if (!degreeSet) {
			throw new IllegalStateException("Degree of polynomial has not been set");
		}
		
		// throws exception if regression can not be performed
		if (xValues.length < 2 | yValues.length < 2) {
			throw new IllegalArgumentException("Must have more than two values");
		}
		
		// fits curve to data
		cost(coefficients, xValues, yValues, lambda);
		
		// set the computed flag to true after we have calculated the coefficients
		computed = true;
	}

	/**
	 * Evaluate the computed model at a certain point
	 * @param x The point to evaluate at
	 * @return The value of the fitted straight line at the point x
	 * 
	 * @see RegressionModel#evaluateAt(double)
	 */
	@Override
	public double evaluateAt(double x) {
		if (!degreeSet) {
			throw new IllegalStateException("Degree of polynomial has not been set and model has not yet computed");
		}
		else if (degreeSet && !computed) {
			throw new IllegalStateException("Model has not yet computed");
		}
		
		double value = 0;
		for(int i=0; i<coefficients.length; i++) {
			value += coefficients[i]*Math.pow(x, i); // check 
		}
		return value;
	}
	
	private double cost(double[] coefficients, double[] x, double[] y) {
		return this.cost(coefficients, x, y, 0.0);
	}
	
	private double cost(double[] coefficients, double[] x, double[] y, double lambda) {
		int m = x.length; // number of training examples
		
		double errorTerm = 0;
		for(int i=0; i<m; i++) {
			errorTerm += Math.pow(hypothesis(x[i], coefficients) - y[i], 2);
		}
		
		double regularizationTerm = 0;
		for(int i=1; i<coefficients.length; i++) {
			regularizationTerm += Math.pow(coefficients[i], 2);
		}
		
		double cost = (.5 / m) * (errorTerm + lambda * regularizationTerm);
		return cost;
	}
	
	private double hypothesis(double x, double[] coefficients) {
		double h = 0;
		for(int i=0; i<coefficients.length; i++) {
			h += coefficients[i] * Math.pow(x, i);
		}
		return h;
	}

}
