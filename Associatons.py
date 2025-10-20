from AssociatorUtils import AssociationState, AssociationObject

# >------------------------------------------------------------------------
#   Class       :   Associator
#   Method      :   Maintain a temporal associations list with measurements
#   Author      :   Brett O'Connor
# >------------------------------------------------------------------------

class Associatons:

    # constructor
    def __init__(self):
        self.associationsList_ = []
        self.latestId_ = -1;

    # function: newAssociation
    # inputs:
    #   - x: the location of the association
    #   - p: the initial covariance of the association
    def newAssociation(self, x, p):
        self.latestId_ += 1
        newAssociationObject = AssociationObject(self.latestId_, x, p)
        self.associationsList_.append(newAssociationObject)

    # TODO - need converged criteria
    # TODO - need closing an association criteria
    # function: updateAssociations
    def updateAssociations(self):

    # function: determineAssociation
    # inputs:
    #   - associationObject
    #   - measurementX: measurement location
    # output: boolean (True, False) to determine whether there is an association or not
    def determineAssociation(associationObject, measurementX):
        # TODO - determine this via mahalanobis distance


    # GETTERS

    # function: getAssociationsByState
    # inputs:
    #   - state: The state (OPEN, CONVERGED, CLOSED) of the associations that are desired
    # outputs:
    #   - List of associations corresponding to the state
    def getAssociationsByState(self, state):
        return [assoc if (assoc.state_ == state) for assoc in self.associationsList_]
    
    # TODO - add a getter function to get association with two states (ex: get open and converged associations)
    
    # function: getAssociationById
    # inputs:
    #   - id: The id desired
    # outputs:
    #   - The association corresponding to the id (returns None if the association does not exist)
    def getAssociationById(self, id):
        targetAssociationList = [assoc if (assoc.id_ == id) for assoc in self.associationsList_]
        if len(targetAssociationList) != 1:
            return None
        return targetAssociationList[0]
