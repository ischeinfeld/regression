package regression;

import java.io.FileReader;
import java.io.IOException;

import com.opencsv.CSVReader;

public class TestPolynomialRegressionModel {

	public static String FILENAME = "./data/regression-data/training_data.txt";

	public static void main(String[] args) throws IOException {
		double[] x;
		double[] y;

		// Initialize
		int numPoints = readNRows(FILENAME);
		System.out.println(String.format("numPoints = %d", numPoints));
		x = new double[numPoints];
		y = new double[numPoints];

		// Load file
		CSVReader reader = new CSVReader(new FileReader(FILENAME));
		String[] nextLine;
		for (int k = 0; k < numPoints; k++) {
			nextLine = reader.readNext();
			x[k] = Float.parseFloat(nextLine[0]);
			y[k] = Float.parseFloat(nextLine[1]);
		}
		reader.close();

		System.out.println("Loaded data.");

		PolynomialRegressionModel model = new PolynomialRegressionModel(x, y);
		model.setDegree(6);
		model.compute();
		double[] coefficients = model.getCoefficients();

		for(double coef : coefficients) {
			System.out.printf("%.4f\n", coef);
		}
	}
	
	protected static int readNRows(String filename) throws IOException{
		CSVReader reader = new CSVReader(new FileReader(filename));
		int n = 0;
		while(reader.readNext() != null){
			n++;
		}
		reader.close();
		return n;
	}
    
	protected static int readNColumns(String filename) throws IOException{
		CSVReader reader = new CSVReader(new FileReader(filename));
		String [] line = reader.readNext();
		reader.close();
		int n = line.length;
		return n;
	}
}
