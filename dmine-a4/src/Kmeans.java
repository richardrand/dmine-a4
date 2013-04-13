import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Comparator;
import java.util.Collections;
import java.util.Random;
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
	
	public Record(int numAttributes) {
		attributes = new double[numAttributes];
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
			System.out.printf("Train stats: %d instances, %d attributes, %d classes, %f entropy\n", instances.size(), instances.get(0).attributes.length, classCount, entropy(instances));
			
			for(DistanceMetric metric : metrics) 
				for(int k = 1; k <= 1; k++)
					runKmeans(instances, k*classCount, metric);

		} catch (FileNotFoundException e) {
			System.err.println("File not found");
			System.exit(EX_NOINPUT);
		}
	}

	static void runKmeans(ArrayList<Record> instances, int k, DistanceMetric metric) {
		System.out.println("ghfthfyh");
		//I'm writing a lot that I feel is redundant
		//might be smarter to write a method that does the clustering 
		//and just do the initial clustering here and then run the reclustering method
		
		int rand = 0;
		Record randomSelection;
		
		//we will be able to treat the centroids as records
		Record[] centroids = new Record[k];
		
		//this will hold an instance's distances to each centroid
		double[] thisInstanceDist = new double[k];
		
		//this will hold all of the clusters
		ArrayList<Record>[] clusters = (ArrayList<Record>[])new ArrayList[k];
		
		Random rnd = new Random();
		//creates k initial centroids
		for (int i = 0; i < k; i++){
			//pick a random index inside of the array
			rand = rnd.nextInt(instances.size());
			
			//choosing initial centroids from data might not be smart, but I think it's the only way
			//since we don't know what the attributes look like
			//--actually i think that's how kmeans is normally done anyway
			randomSelection = instances.get(rand);
			System.out.println(randomSelection);
			//adds the selection to the list of centroids
			centroids[i] = randomSelection;
			//adds a new cluster to the list of clusters
			clusters[i] = new ArrayList<Record>();
			//adds the centroid to that cluster
			//the following step will be done automatically  in the kmeans loop
			//clusters.get(i).add(randomSelection);
		}
		
		//this is the actual kmeans algorithm
		//runs while centroids are changing
		//checks if all old centroids are same as new centroids, if so false
		//will run once more
		//for each instance, measures distances to each centroid
		//chooses closest centroid and adds the instance to that cluster
		//stores old centroids in an ArrayList
		//creates new centroids by some sort of means measure
		//reiterate
		boolean centroids_keep_changing = true;
		while(centroids_keep_changing) {
			System.out.println("go");
			centroids_keep_changing = false; //assume centroids haven't changed. later, check each centroid as it's updated and set this flag if one has changed.
			
			for(ArrayList<Record> cluster : clusters)
				cluster.clear(); //clear out the old clusters before we reassign every instance
				
			for(Record instance : instances) {
				double shortest_dist = 0;
				int shortest_dist_i = -1;
				//for each instance, measures distances to each centroid
				for(int i = 0; i < centroids.length; i++) {
					double dist = metric.distanceBetween(instance, centroids[i]);
					if(Double.isNaN(dist)) {
						System.out.println(centroids[i]);
						System.exit(0);
					}
					if(shortest_dist_i == -1 || dist < shortest_dist) {
						shortest_dist = dist;
						shortest_dist_i = i;
					}
					thisInstanceDist[i] = dist; 
				}
				//chooses closest centroid and adds the instance to that cluster
				clusters[shortest_dist_i].add(instance);
			}
			
			//creates new centroids by some sort of means measure
			for(int i = 0; i < k; i++) {
				for(int a = 0; a < instances.get(0).attributes.length; a++) {
					double sum_a = 0;
					for(Record r : clusters[i])
						sum_a += r.attributes[a];
						
					//creates new centroids by some sort of means measure
					if(centroids[i].attributes[a] != sum_a/clusters[i].size()) {
						centroids_keep_changing = true;
						centroids[i].attributes[a] = sum_a/clusters[i].size();
					}
					//System.out.println(sum_a);
					//System.out.println("Centroid " + i + centroids[i]);
				}
				
				System.out.println(entropy(clusters[i]));
			}
		}
	
		
		
	}
	
	static double entropy(ArrayList<Record> cluster) {
		double result = 0;
		System.out.print("matches for this cluster per class: ");
		for(String className : Kmeans.classes.keySet()) {
			int matches = 0;
			for(Record instance : cluster)
				if(instance.classname.equals(className))
					matches++;
			double x = ((double)matches)/cluster.size();
			result -= x*lg(x);
			System.out.print(matches + " ");
		}
		System.out.println();
		return result;
	}
	
	//needed this for the BSS. Dunno if it's really necessary, 
	//but since the centroid averaging was couched in the runKmeans loop
	//I needed to pull something similar out.
	static Record dataMidpoint(ArrayList<Record> instances){
		
		double sum_a;
		Record result = new Record(instances.get(0).toString());
		//there might be a more optimized way to create the result
		//I just wanted to make sure changing the result wasn't changing anything
		//in the instances array
		
		for (int a = 0; a < instances.get(0).attributes.length; a++){
			sum_a = 0;
			for (Record r : instances){
				sum_a += r.attributes[a];
			}
			result.attributes[a] = sum_a/instances.size();
		}
		
		return result;
	}
	
	static double WSS(ArrayList<ArrayList> clusters, ArrayList<Record> midpoints, DistanceMetric metric){
		
		double sum1, sum2 = 0;
		
		System.out.print("WSS for this clustering: ");
		//for the ith cluster
		for (int i = 0; i < clusters.size(); i++){
			//for each record in that cluster
			for (Record r : clusters.get(i)){
				//sum the square of the distance between record and the cluster's midpoint
				sum1 += metric.distanceBetween(r,midpoints.get(i)) * metric.distanceBetween(r,midpoints.get(i));
			}
			sum2 += sum1;
		}
		
		System.out.print(sum2 + ".");
		System.out.println();
		
		return sum2;
	}
	
	static double BSS(ArrayList<ArrayList> clusters, ArrayList<Record> midpoints, DistanceMetric metric){
		
		double sum = 0;
		double size = 0;
		Record midpoint; 
		
		System.out.print("BSS for this clutering: ");
		//for the ith cluster
		for (int i = 0; i < clusters.size(); i++){
			
			size = clusters.get(i).size();
			midpoint = dataMidpoint(clusters.get(i));
			//sum the product of the size and the squared distance between the cluster's midpoint and the absolute midpoint
			sum1 += size * metric.distanceBetween(midpoints.get(i),midpoint) * metric.distanceBetween(midpoints.get(i),midpoint);
		}
		
		System.out.print(sum + ".");
		System.out.println();
		return sum;
	}
	
	static double lg(double x) {
		if(x == 0) return 0;
		return Math.log(x)/Math.log(2);
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