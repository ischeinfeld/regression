package regression;

public class TestLinearRegressionModel {
	
	public static void main(String[] args) {
		double[] x = { 2, 3, 4, 5, 6, 8, 10, 11 };

		double[] y = { 21.05, 23.51, 24.23, 27.71, 30.86, 45.85, 52.12, 55.98 };

		System.out.println("Expected output from Excel: y = 9.4763 + 4.1939x");

		RegressionModel model = new LinearRegressionModel(x, y);
		model.compute();
		double[] coefficients = model.getCoefficients();

		System.out.printf("Actual output from our code: y = %.4f + %.4fx", coefficients[0], coefficients[1]);
	}
}
