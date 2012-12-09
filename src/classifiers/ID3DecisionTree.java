/**
 * 
 */
package classifiers;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Vector;

import common.Classifier;
import common.ClassifierException;
import common.CommonUtils;
import common.CommonUtils.EntropyMeasure;
import common.DecisionTree;
import common.FeatureVector;
import common.FeatureVectorException;
import common.TrainingException;

/**
 * @author jaison
 *
 */
public class ID3DecisionTree extends Classifier {

	DecisionTree decisionRoot;
	EntropyMeasure e;
	
	/**
	 * 
	 */
	public ID3DecisionTree() {
		_classifierName = "ID3DecisionTree";
		e = EntropyMeasure.INFO_GAIN;
	}

	public void setEntropyMeasure(EntropyMeasure em) {
		e = em;
	}

	/* (non-Javadoc)
	 * @see common.Classifier#train()
	 */
	@Override
	public boolean train() throws TrainingException, ClassifierException {
		// TODO Auto-generated method stub
		decisionRoot = learnDecisionTree(_trainingVector);
		return false;
	}

	/* (non-Javadoc)
	 * @see common.Classifier#predictClass(common.FeatureVector)
	 */
	@Override
	protected String predictClass(FeatureVector fv) throws TrainingException,
			ClassifierException, FeatureVectorException {
		if(decisionRoot == null)
			throw new ClassifierException("ID3 Exception : NULL decision tree. Forgot to train ?");
		
		System.out.println(fv.getFeatureCSVString(","));
		
		DecisionTree curNode = decisionRoot;
		while(!curNode.isLeafNode()) {

			if(curNode.isNominalNode()) {
				int featureIdx = curNode.getNominalFeatureIndex();
				String nomValue = fv.getCategoricalValues().get(featureIdx);
			
				curNode = curNode.getNominalBranch(nomValue);
			}
			else {
				int featureIdx = curNode.getRealFeatureIndex();
				Double realValue = fv.getRealValues().get(featureIdx);
				
				curNode = curNode.getRealBranch(realValue);
			}

			if(curNode == null)
				throw new ClassifierException("ID3 Exception : No branch to continue");

		}
	
		//System.out.println("Predicting class as "+curNode.getClassLabel());
		return curNode.getClassLabel();
	}

	private DecisionTree learnDecisionTree(Vector<FeatureVector> vf) throws ClassifierException {

		DecisionTree decisionNode = null;
		
		if(vf.size() == 0 ) {
			throw new ClassifierException("ID3 Exception : Cannot form decision tree node on nullSet");
		}

		double curEntropy = CommonUtils.getEntropy(vf);
		if(curEntropy == 0) {
			decisionNode = new DecisionTree(vf.get(0).getClassification());
			return decisionNode;
		}

		if(vf.get(0).hasRealValues()) {
			int realFeatueCount = vf.get(0).getRealValues().size();
			for(int i=0;i<realFeatueCount;i++) {
				DecisionTree tmpNode = getDecisionNodeForRealFeature(vf, i);
				if(decisionNode == null || decisionNode.getGain() < tmpNode.getGain() )
					decisionNode = tmpNode;
			}
		}
		
		if(vf.get(0).hasCategoricalValues()) {
			int nomFeatureCount = vf.get(0).getCategoricalValues().size();
			for(int i=0;i<nomFeatureCount;i++) {
				DecisionTree tmpNode = getDecisionNodeForNominalFeature(vf, i);
				if(decisionNode == null || decisionNode.getGain() < tmpNode.getGain() )
					decisionNode = tmpNode;
				}
			}
		
		if(decisionNode.getGain() == 0.0) {
			String majorityVote = CommonUtils.getMajorityVote(vf);

			decisionNode = new DecisionTree(majorityVote);
			return decisionNode;
		}
		
		if(decisionNode.isNominalNode()) {
			/* Get buckets of children and their corresponding trees */
			int featureIdx = decisionNode.getNominalFeatureIndex();
			HashMap<String, Vector<FeatureVector>> childBuckets = new HashMap<String, Vector<FeatureVector>>();
			HashMap<String, DecisionTree> childNodes = new HashMap<String, DecisionTree>();

			for(int i=0;i<vf.size();i++) {
				FeatureVector curSample = vf.get(i);
				String curNominalValue = curSample.getCategoricalValues().get(featureIdx);
				
				Vector<FeatureVector> curBucket = childBuckets.get(curNominalValue);
				if(curBucket == null ){
					curBucket = new Vector<FeatureVector>();
					childBuckets.put(curNominalValue, curBucket);
				}
				
				curBucket.add(curSample);
			}
			
			DecisionTree mlBranch = null;
			int maxBucketSize = 0;
			
			for(Map.Entry<String, Vector<FeatureVector>> bucket : childBuckets.entrySet()) {
				String nomValue = bucket.getKey();
				Vector<FeatureVector> vfChild = bucket.getValue();
				
				DecisionTree childNode = learnDecisionTree(vfChild);
				childNodes.put(nomValue, childNode);
				
				if(vfChild.size() > maxBucketSize) {
					maxBucketSize = vfChild.size();
					mlBranch = childNode;
				}
			}

			decisionNode.setNominalChildMap(childNodes);
			decisionNode.setMostLikelyChild(mlBranch);			
		}
		else { 
			int featureIdx = decisionNode.getRealFeatureIndex();
			double splitPoint = decisionNode.getRealFeatureSplitPoint();
			
			Vector<FeatureVector> leftSubTree = new Vector<FeatureVector>(); 
			Vector<FeatureVector> rightSubTree = new Vector<FeatureVector>(); 
			Vector<FeatureVector> unknownSubTree = new Vector<FeatureVector>(); 

			for(int i=0;i<vf.size();i++) {
				FeatureVector curSample = vf.get(i);
				Double realFeatureValue = curSample.getRealValues().get(featureIdx);
				
				if(realFeatureValue == null)
					unknownSubTree.add(curSample);
				else if(realFeatureValue < splitPoint) 
					leftSubTree.add(curSample);
				else
					rightSubTree.add(curSample);
			}
			
			if(leftSubTree.size() > rightSubTree.size()) 
			{
				leftSubTree.addAll(unknownSubTree);
			}
			else {
				rightSubTree.addAll(unknownSubTree);				
			}

			DecisionTree leftChild = learnDecisionTree(leftSubTree);
			DecisionTree rightChild = learnDecisionTree(rightSubTree);
			decisionNode.setSubTrees(leftChild, rightChild);
			
			if(leftSubTree.size() > rightSubTree.size())
				decisionNode.setMostLikelyChild(leftChild);
			else 
				decisionNode.setMostLikelyChild(rightChild);				
		}
		
		return decisionNode;
	}
	
	private DecisionTree getDecisionNodeForNominalFeature(Vector<FeatureVector> vf, int featureIdx) throws ClassifierException {

			DecisionTree decisionNode = null;
			
			if(vf.size() == 0 ) {
				throw new ClassifierException("ID3 Exception : Cannot form decision tree node on nullSet");
			}
			
			double curEntropy = CommonUtils.getEntropy(vf);
			if(curEntropy == 0) {
				decisionNode = new DecisionTree(vf.get(0).getClassification());
			}
			else {
				double gain = CommonUtils.getInformationGainOnNominalFeature(vf, featureIdx, e);
				decisionNode = new DecisionTree(featureIdx);
				decisionNode.setEntropy(curEntropy);
				decisionNode.setGain(gain);
			}			
			return decisionNode;
	}

	private DecisionTree getDecisionNodeForRealFeature(Vector<FeatureVector> vf, int featureIdx) throws ClassifierException {

		DecisionTree decisionNode = null;
		
		if(vf.size() == 0 ) {
			throw new ClassifierException("ID3 Exception : Cannot form decision tree node on nullSet");
		}
		
		double curEntropy = CommonUtils.getEntropy(vf);
		if(curEntropy == 0) {
			decisionNode = new DecisionTree(vf.get(0).getClassification());
		}
		else {
			/* Try all splitPoints */
			HashSet<Double> possibleSplitPoints = new HashSet<Double>();
			
			for(int i=0;i<vf.size();i++) {
				Double curVal = vf.get(i).getRealValues().get(featureIdx);
				if(curVal != null)
					possibleSplitPoints.add(curVal);
			}
			
			double bestGain = 0, bestSplitPoint = -1;
			
			Iterator itr = possibleSplitPoints.iterator();
			while(itr.hasNext()) {
				double curSplitPoint = (Double) itr.next();
				double gain = CommonUtils.getInformationGainOnRealFeature(vf, featureIdx, curSplitPoint, e);
				
				if(gain > bestGain) {
					bestGain = gain;
					bestSplitPoint = curSplitPoint;
				}
			}

			decisionNode = new DecisionTree(bestSplitPoint, featureIdx);
			decisionNode.setEntropy(curEntropy);
			decisionNode.setGain(bestGain);
		}			
		return decisionNode;
}

	public void displayDebugInfo() {
		System.out.println(decisionRoot.getTree(0));
	}

	public EntropyMeasure getEntropyMeasure() {
		return e;
	}

}
