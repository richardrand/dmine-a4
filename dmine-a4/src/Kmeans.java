import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Comparator;
import java.util.Collections;
import java.text.DecimalFormat;
import javax.swing.JPanel;
import java.awt.GridLayout;

class Class {};

//Parses and stores train and test records, and predicts the class of test records with the help of a DistanceMetric.
class Record {
	double[] attributes;
	String classname;
	
	Record(String line) {
		String[] fields = line.split(",");
		
		attributes = new double[fields.length - 1];
		for(int i = 0; i < fields.length - 1; i++)
			attributes[i] = (double)Double.parseDouble(fields[i]);

		//Globally keep track of class names so we can properly form confusion matrices, and so we know how many graphs to plot
		classname = fields[fields.length - 1];
		if(!Kmeans.classes.containsKey(classname))
			Kmeans.classes.put(classname, new Class());
		
		assert(attributes.length == fields.length - 1);
	}
	
	//Used for debugging.
	public String toString() {
		String result = "";
		for(int i = 0; i < 5 && i < attributes.length; i++)
			result += String.format("%10.5f", attributes[i]);
		result += "..." + String.format("%10.5f", attributes[attributes.length-1]) + " " + classname;
		return result;
	}
	
	//Called by DistanceMetric to predict class using kNN.
	String predictClass(ArrayList<Record> trainingRecords, DistanceMetric how, int k) {
		//Wrap each training record up in this object in order to find the closest k records
		class RecordWithDistance implements Comparable<RecordWithDistance> {
			RecordWithDistance(Record r, double d) {trainingRec = r; distance = d;}
			Record trainingRec;
			double distance;
			public int compareTo(RecordWithDistance other) {
				if(distance < other.distance) return -1;
				else if(distance == other.distance) return 0;
				else return 1;
			}
		}
		RecordWithDistance[] recsWithDist = new RecordWithDistance[trainingRecords.size()];

		//Compute and store distance to each record
		for(int i = 0; i < trainingRecords.size(); i++)
			recsWithDist[i] = new RecordWithDistance(trainingRecords.get(i), how.distanceBetween(this, trainingRecords.get(i)));
		
		//Sort the closest records first, so the first k records in list are the closest ones.
		java.util.Arrays.sort(recsWithDist); 
		
		//Hold the classes found in the nearest k samples in a hash table, using the values to hold the weighted vote of each class on the final prediction
		HashMap<String, Double> closeClasses = new HashMap<String, Double>();
		for(int i = 0; i < k && i < recsWithDist.length; i++) {
			String foundClass = recsWithDist[i].trainingRec.classname;
			if(!closeClasses.containsKey(foundClass)) closeClasses.put(foundClass, 0.0);
			closeClasses.put(foundClass, closeClasses.get(foundClass) + 1/recsWithDist[i].distance); //Increase the weight of this class by the inverse of the distance to the neighbor
		}

		//Return the class with the greatest weight. TODO: Consider switching to priority queue/max heap
		
		Comparator<Map.Entry<String, Double>> findClosestClass = new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> a, Map.Entry<String, Double> b) {
				return a.getValue().compareTo(b.getValue());
			}
		};

		return Collections.max(closeClasses.entrySet(), findClosestClass).getKey();
	}
}
	
abstract class DistanceMetric {
	//Polymorphism allows the code organization to be simplified, and makes it more extensible.
	public abstract double distanceBetween(Record a, Record b);		
}

class Euclidean extends DistanceMetric {
	public double distanceBetween(Record a, Record b) {
		assert(a.attributes.length == b.attributes.length);
		
		double dist = 0;
		for (int attrib = 0; attrib < a.attributes.length; attrib++) {
			double temp = (a.attributes[attrib] - b.attributes[attrib]);
			dist += temp*temp;
		}
		return Math.sqrt(dist); //dist
	}
}

class Cosine extends DistanceMetric {
	public double distanceBetween(Record a, Record b) {
		assert(a.attributes.length == b.attributes.length);
		
		double ab_dot_product = 0, a_magnitude_squared = 0, b_magnitude_squared = 0;
		for (int attrib = 0; attrib < a.attributes.length; attrib++) {
			ab_dot_product += a.attributes[attrib] * b.attributes[attrib];
			a_magnitude_squared += a.attributes[attrib] * a.attributes[attrib];
			b_magnitude_squared += b.attributes[attrib] * b.attributes[attrib];
		}
		double cossim = ab_dot_product/(Math.sqrt(a_magnitude_squared)*Math.sqrt(b_magnitude_squared));
		return 1 - cossim;
	}
}

public class Kmeans {
	//Error status for when the input file given is not available.
	static final int EX_NOINPUT = 66;
	
	public static HashMap<String, Class> classes = new HashMap<String, Class>();
	//Store an array of distance metrics for enumeration
	static final DistanceMetric[] metrics = new DistanceMetric[] {
		new Euclidean(),
		new Cosine()
	};
	
	static ArrayList<Record> parseArff(String fileName) throws IOException {
		ArrayList<Record> records = new ArrayList<Record>();
			
		//@data is the line immediately preceding csv
		BufferedReader inputStream = new BufferedReader(new FileReader(fileName));
		while (inputStream.readLine().indexOf("@data") == -1);
		
		String line;
		while ((line = inputStream.readLine()) != null && line.indexOf(",") != -1)
			records.add(new Record(line));
		
		return records;
	}
	
	public static void main(String[] args) throws IOException {
		System.out.print("Type i to run on (i)ris, a to run on (a)ll genes, s for (s)ignificant genes, or return to enter a custom pair of filenames: ");
		String test = System.console().readLine();
		if(test.equals("i")) {
			cluster("iris.arff");
		} else if(test.equals("a")) {
			cluster("AllGenes.arff");
		} else if(test.equals("s")) {
			cluster("SigGenes.arff");
		} else {
			System.out.println("Name of input file: ");
			cluster(System.console().readLine());
		}
	}
	
	static void cluster(String file) throws IOException {
		try {
			System.out.println("Clustering " + file);
			
			//Parse files
			ArrayList<Record> instances = parseArff(file);
			int classCount = classes.size();
			System.out.printf("Train stats: %d instances, %d attributes, %d classes\n", instances.size(), instances.get(0).attributes.length, classCount);
			
			for(DistanceMetric metric : metrics) 
				for(int k = 1; k++; k <= 3)
					runKmeans(instances, k*classCount, metric);
			
			//Print tables to stdout
			printTables();
			
		} catch (FileNotFoundException e) {
			System.err.println("File not found");
			System.exit(EX_NOINPUT);
		}
	}

	static void runKmeans(ArrayList<Record> instances, int k, DistanceMetric metric) {
	
	}

	//Loop through the data series and print them out in csv form.
	static void printTables() {
		for(Map.Entry<String, Class> className : classes.entrySet()) {
			String precisionTables = "", recallTables = "", F1Tables = "";
			System.out.println(className.getKey());

			for(int k = 0; k < 5; k++) {
				precisionTables += "k=" + (k*2+3) + ",";
				recallTables += "k=" + (k*2+3) + ",";
				F1Tables += "k=" + (k*2+3) + ",";
				for(DistanceMetric dm : metrics) {
					precisionTables += className.getValue().precision.getSeries(dm.getClass().getName().replaceAll("_"," ")).getY(k) + ",";
					recallTables += className.getValue().recall.getSeries(dm.getClass().getName().replaceAll("_"," ")).getY(k) + ",";
					F1Tables += className.getValue().Fmeasure.getSeries(dm.getClass().getName().replaceAll("_"," ")).getY(k) + ",";
				}
				precisionTables += "\n";
				recallTables += "\n";
				F1Tables += "\n";
			}
			
			System.out.println("precision");
			System.out.println(precisionTables);
			System.out.println("recall");
			System.out.println(recallTables);
			System.out.println("F1");
			System.out.println(F1Tables);
		}
	}
	
	//The following methods can be used if normalization is necessary.
	static void altMinMaxNorm(ArrayList<Record> tests, ArrayList<Record> trains) {
		int numAttribs = tests.get(0).attributes.length;
		assert(trains.get(0).attributes.length == numAttribs);
		
		for(int i = 0; i < numAttribs; i++) {
			double min = 20, max = 16000;
			double range = max - min;
			for(Record r : trains) {
				if(range == 0) r.attributes[i] = 0.5;
				else r.attributes[i] =  (r.attributes[i] - min)/range;

				assert((r.attributes[i] >= 0.0) && (r.attributes[i] <= 1.0));
			}
			for(Record r : tests) {
				if(range == 0) r.attributes[i] = 0.5;
				else r.attributes[i] =  (r.attributes[i] - min)/range;

				assert((r.attributes[i] >= 0.0) && (r.attributes[i] <= 1.0));
			}
		}
	}
	
	static void minMaxNorm(ArrayList<Record> tests, ArrayList<Record> trains) {
		int numAttribs = tests.get(0).attributes.length;
		assert(trains.get(0).attributes.length == numAttribs);
		
		for(int i = 0; i < numAttribs; i++) {
			double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
			for(Record r : trains) {
				if(r.attributes[i] <= min) min = r.attributes[i];
				if(r.attributes[i] >= max) max = r.attributes[i];
			}
			for(Record r : tests) {
				if(r.attributes[i] <= min) min = r.attributes[i];
				if(r.attributes[i] >= max) max = r.attributes[i];
			}
			double range = max - min;
			for(Record r : trains) {
				if(range == 0) r.attributes[i] = 0.5;
				else r.attributes[i] =  (r.attributes[i] - min)/range;

				assert((r.attributes[i] >= 0.0) && (r.attributes[i] <= 1.0));
			}
			for(Record r : tests) {
				if(range == 0) r.attributes[i] = 0.5;
				else r.attributes[i] =  (r.attributes[i] - min)/range;

				assert((r.attributes[i] >= 0.0) && (r.attributes[i] <= 1.0));
			}
		}
	}
	
	//Doesn't work :(
	static void zscoreNorm(ArrayList<Record> tests, ArrayList<Record> trains) {
		int numAttribs = tests.get(0).attributes.length;
		assert(trains.get(0).attributes.length == numAttribs);
		
		for(int i = 0; i < numAttribs; i++) {
			double sum = 0;
			for(Record r : trains) sum += r.attributes[i];
			double mean = sum / trains.size();
			sum = 0;
			for(Record r : trains) sum += (r.attributes[i] - mean)*(r.attributes[i] - mean);
			double stddev = Math.sqrt(sum/(trains.size() - 1));
			
			for(Record r : trains)
				r.attributes[i] = (r.attributes[i] - mean)/stddev;
			for(Record r : tests)
				r.attributes[i] = (r.attributes[i] - mean)/stddev;
		}
	}
}