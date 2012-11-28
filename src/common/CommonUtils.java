package common;

import java.util.Vector;

public class CommonUtils {

	public static double getEuclidianDistance(FeatureVector f1, FeatureVector f2) 
			throws FeatureVectorException {
		double euclidianDis = 0;
		
		if(f1.checkCompatibility(f2) == false)
			throw new FeatureVectorException("FeatureVectors passed are " +
					"incompatible with each other for distance measurement");
		
		if(f1.hasCategoricalValues())
			throw new FeatureVectorException("Euclidian Distance " +
					"cannot be computed on feature Vectors with nominal values");

		if(f1.hasRealValues())
		{
			Vector<Double> realValues1 = f1.getRealValues();
			Vector<Double> realValues2 = f2.getRealValues();
		
			for(int i=0;i<realValues1.size();i++)
			{
				double d1 = realValues1.elementAt(i);
				double d2 = realValues2.elementAt(i);
				euclidianDis += Math.sqrt((d1-d2)*(d1-d2));
			}
		}
		
		if(f1.hasNumericalValues())
		{
			Vector<Integer> numValues1 = f1.getNumericalValues();
			Vector<Integer> numValues2 = f2.getNumericalValues();
		
			for(int i=0;i<numValues1.size();i++)
			{
				double d1 = numValues1.elementAt(i);
				double d2 = numValues2.elementAt(i);
				euclidianDis += Math.sqrt((d1-d2)*(d1-d2));
			}
		}
		
		return euclidianDis;
	}
}
