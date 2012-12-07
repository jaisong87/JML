package common;

import java.util.Map;

public class DecisionTree {

	/* Splitting on nominal feature here*/
	int nominalFeatureIdx;
	Map<String, DecisionTree> childNodes;  
	
	/* Splitting on real features */
	int realFeatureIdx;
	double realFeatureSplitPoint;
	DecisionTree leftSubTree, rightSubTree;
	
	double entropy, gain; /* stats on the entropy here */
	
	/* Applicable for leafNodes */
	boolean isLeaf, isNominalNode;
	String classificationLabel; 
	
	/* Internal node based on nominal feature */
	public DecisionTree( int featureIdx) {
		nominalFeatureIdx = featureIdx;
		isNominalNode = true;
	}
	
	/* Internal node that splits a real valued feature */
	public DecisionTree(double splitPoint, int featureIdx) {
		realFeatureIdx = featureIdx;
		realFeatureSplitPoint = splitPoint;
		isNominalNode = false;
	}

	/* Leaf node for a decision tree */
	public DecisionTree(String classLabel) {
		classificationLabel = classLabel;
		isLeaf = true;
		isNominalNode = false;
	}
	
	public void setEntropy(double e) {
		entropy = e;
	}
	
	public void setGain(double g) {
		gain = g;
	}
	
	public void setNominalChildMap(Map<String, DecisionTree> children) {
		childNodes = children;
	}

	public void setSubTrees(DecisionTree leftTree, DecisionTree rightTree) {
		leftSubTree = leftTree;
		rightSubTree = rightTree;
	}
	
	public double getGain() {
		return gain;
	}

	public double getEntropy() {
		return entropy;
	}

	public boolean isNominalNode() {
		return isNominalNode;
	}

	public int getNominalFeatureIndex() {
		return nominalFeatureIdx;
	}
	
	public int getRealFeatureIndex() {
		return realFeatureIdx;
	}
	
	public double getRealFeatureSplitPoint() {
		return realFeatureSplitPoint;
	}
	
	public String getTree(int level) {
		String treeExpr = "";
		for(int i=0;i<level;i++)
			treeExpr+="|   ";
		
		/* Write Logic later*/
		if(isLeaf) {
			treeExpr += "CLASS="+classificationLabel+"\n";
		}
		
		else if(isNominalNode) {
			treeExpr += "Split On NOM : "+nominalFeatureIdx+"("+childNodes.size()+" children)\n";
			
			for( Map.Entry<String, DecisionTree>child : childNodes.entrySet())
				treeExpr += child.getKey() + " " + child.getValue().getTree(level+1);
		}
		else {
			treeExpr += "Split on Real : " + realFeatureIdx+" < " + realFeatureSplitPoint+" => "+leftSubTree.getTree(level+1)+"\n";				
			treeExpr += "Split on Real : " + realFeatureIdx+" >= " + realFeatureSplitPoint+" => "+rightSubTree.getTree(level+1)+"\n";				
		}
		
		return treeExpr;
	}

}
