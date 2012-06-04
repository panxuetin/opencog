#ifndef _SPATIAL_OCTREE3DMAPMANAGER_H
#define _SPATIAL_OCTREE3DMAPMANAGER_H

#include <map>
#include <vector>
#include "Block3DMapUtil.h"
#include "Block3D.h"
#include "Octree.h"
#include <opencog/atomspace/Handle.h>

using namespace std;

namespace opencog
{
    namespace spatial
    {
        class Entity3D;
        class BlockEntity;
        class Octree;

        enum SPATIAL_RELATION
        {
            LEFT_OF = 0,
            RIGHT_OF,
            ABOVE,
            BELOW,
            BEHIND,
            IN_FRONT_OF,
            BESIDE,
            NEAR,
            FAR_,
            TOUCHING,
            BETWEEN,
            INSIDE,
            OUTSIDE,
            ON_TOP_OF,
            ADJACENT,

            TOTAL_RELATIONS
        };

        class Octree3DMapManager
        {
        public:

            // to store the blockEntities's node handles just diasppear,
            // the ~Blockentity() will add its Handel into this list, DO NOT add to this list from other place
            static vector<Handle>  newDisappearBlockEntityList;

            // to store the blockEntities just appear
            // the Blockentity() will add itself into this list, DO NOT add to this list from other place
            static vector<BlockEntity*> newAppearBlockEntityList;

            // to store the blockentities need to be updated the predicates
            static vector<BlockEntity*> updateBlockEntityList;

            static const int AccessDistance = 2;

            /**
             * @ min_x and min_y is the start position of this octree space
             * @_floorHeight: the height of the floor, the z value of  start position
             * @_offSet: how many unit per edge in this space, indicating the size of the whole space
             */
            Octree3DMapManager(std::string _mapName, int _xMin, int _yMin, int _zMin, int _xDim, int _yDim, int _zDim, int _floorHeight);
            ~Octree3DMapManager(){};

            bool hasPerceptedMoreThanOneTimes;

            const map<Handle, BlockVector>& getAllUnitBlockatoms(){return mAllUnitBlockAtoms;}

            const vector<BlockEntity*>& getBlockEntityList(){return mBlockEntityList;}

            const map<Handle, Entity3D*>& getAllNoneBlockEntities(){return mAllNoneBlockEntities;}

            int getTotalDepthOfOctree(){return mTotalDepthOfOctree;}

            inline  int   getFloorHeight() const {return mFloorHeight;}
            inline  int   getTotalUnitBlockNum() const {return mTotalUnitBlockNum;}
            inline  const AxisAlignedBox& getMapBoundingBox() const {return mMapBoundingBox;}
            inline  std::string  getMapName()const {return mMapName;}

            inline  int   xMin() const{return mMapBoundingBox.nearLeftBottomConer.x;}
            inline  int   yMin() const{return mMapBoundingBox.nearLeftBottomConer.y;}
            inline  int   zMin() const{return mMapBoundingBox.nearLeftBottomConer.z;}

            inline  int   xMax() const{return mMapBoundingBox.nearLeftBottomConer.x + mMapBoundingBox.size_x;}
            inline  int   yMax() const{return mMapBoundingBox.nearLeftBottomConer.y + mMapBoundingBox.size_y;}
            inline  int   zMax() const{return mMapBoundingBox.nearLeftBottomConer.z + mMapBoundingBox.size_z;}

            inline  int   xDim() const{return mMapBoundingBox.size_x;}
            inline  int   yDim() const{return mMapBoundingBox.size_y;}
            inline  int   zDim() const{return mMapBoundingBox.size_z;}

            // currently we consider all the none block entities has no collision, agents can get through them
            void addNoneBlockEntity(const Handle &entityNode, BlockVector _centerPosition,
                                    int _width, int _lenght, int _height, double yaw, std::string _entityName, bool is_obstacle = false);

            void removeNoneBlockEntity(const Handle &entityNode);

            void addSolidUnitBlock(BlockVector _pos, const Handle &_unitBlockAtom = opencog::Handle::UNDEFINED,  std::string _materialType = "", std::string _color = "" );

            // return the BlockEntity occupied this position, then the atomspace can update the predicates for this Entity
            // But if this entity is disappear during this process, then will return 0
            void removeSolidUnitBlock(const Handle &blockNode);

            // Given a posititon, find all the blocks in the BlockEntity this posititon belongs to.
            BlockEntity* findAllBlocksInBlockEntity(BlockVector& _pos);
            inline const Octree* getRootOctree(){return mRootOctree;}

            // get a random near position for building a same blockEnity as the _entity
            // this position should have enough space to build this new entity,
            // not to be overlapping other objects in the map.
            BlockVector getBuildEnityOffsetPos(BlockEntity* _entity) const;

            // Return the blockEntity occupies this postiton
            // If there is not a blockEntity here, return 0
            BlockEntity* getEntityInPos(BlockVector& _pos) const;

            // this should be call only once just after the map perception finishes at the first time in the embodiment.
            void findAllBlockEntitiesOnTheMap();

            // just remove this entity from the mBlockEntityList, but not delete it yet
            void removeAnEntityFromList(BlockEntity* entityToRemove);

            const Entity3D* getEntity( const Handle entityNode ) const;

            const Entity3D* getEntity( std::string entityName) const;

            std::string getEntityName(Entity3D* entity) const;

            // return the location of the given object handle. This object can be a block,nonblockentity or blockentity
            BlockVector getObjectLocation(Handle objNode) const;

            // return the Direction of the given object face to. This object can be a block,nonblockentity or blockentity
            // Because it make no sense if this object is a block, we just define the directions of all the blocks are all BlockVector::X_UNIT direction (1,0,0)
            // if there is nothting for this handle on this map, return BlockVector::Zero
            BlockVector getObjectDirection(Handle objNode) const;

            BlockEntity* findBlockEntityByHandle(const Handle entityNode) const;

            /**
             * Find a free point near a given position, at a given distance
             * @param position Given position
             * @param distance Maximum distance from the given position to search the free point
             * @param startDirection Vector that points to the direction of the first rayTrace
             * @param toBeStandOn if this is true then agent can stand at that position,which means the point should not be on the sky
             * x
             */
            BlockVector getNearFreePointAtDistance( const BlockVector& position, int distance, const BlockVector& startDirection, bool toBeStandOn = true ) const;

            // check whether people can stand on this position or not, which means first there is not any obstacle or block here and there is a block under it.
            bool checkStandable(BlockVector& pos) const;
            bool checkStandable(int x, int y, int z) const;

            bool containsObject(const Handle objectNode) const;
            bool containsObject(std::string& objectname) const;

            /**
             * TODO: Persistence
             */
            void save(FILE* fp ){};
            void load(FILE* fp ){};

            static std::string toString( const Octree3DMapManager& map );

            static Octree3DMapManager* fromString( const std::string& map );


            template<typename Out>
                 Out findAllEntities(Out out) const
            {

                 // only calculate the non-block entities and block entities, no including the blocks
                 std::vector<const char*> objectNameList;

                 // non-block entities:
                 map<Handle, Entity3D*> ::const_iterator it;

                 for ( it = mAllNoneBlockEntities.begin( ); it != mAllNoneBlockEntities.end( ); ++it )
                 {
                     objectNameList.push_back(getEntityName((Entity3D*)(it->second)).c_str( ));
                 } // for

                 return std::copy(objectNameList.begin(), objectNameList.end(), out);
             }


                 // todo: now we only count the entities, we may need to return all the unit blocks as well
             template<typename Out>
                     Out getAllObjects(Out out) const
             {
                return findAllEntities(out);
             }

            /*
             * Threshold to consider an entity next to another
             * @return Next distance
             */
            inline double getNextDistance( void ) const {
                return 2.0;
            }

            /**
             * Extract the spatial relations between two objects
             *
             * @param observer The observer entity
             * @param besideDistance A distance used as threshold for considering
             *                       an object beside or not another
             * @param entityB The entity used as reference entity
             * @return std::vector<SPATIAL_RELATION> a vector of all spatial relations
             *         between entityA (this entity) and entityB (reference entity)
             *
             */
            std::vector<SPATIAL_RELATION> computeSpatialRelations( const Entity3D* observer,
                                                                   double besideDistance,
                                                                   const Entity3D* entityA,
                                                                   const Entity3D* entityB ) const;

            /**
             * Finds the list of spatial relationships that apply to the three entities.
             * Currently this can only be BETWEEN, which states that A is between B and C
             *
             * @param observer The observer entity
             * @param besideDistance A distance used as threshold for considering
             *                       an object beside or not another
             * @param entityB First reference entity
             * @param entityC Second reference entity
             *
             * @return std::vector<SPATIAL_RELATION> a vector of all spatial relations
             *         among entityA (this entity), entityB (first reference) and entityC
             *         (second reference)
             *
             */
            std::vector<SPATIAL_RELATION> computeSpatialRelations( const Entity3D* observer,
                                                                   double besideDistance,
                                                                   const Entity3D* entityA,
                                                                   const Entity3D* entityB,
                                                                   const Entity3D* entityC ) const;

            /**
             * Return a string description of the relation
             */
            static std::string spatialRelationToString( SPATIAL_RELATION relation );

        protected:

            map<Handle, BlockVector> mAllUnitBlockAtoms;
            vector<BlockEntity*> mBlockEntityList;
            map<Handle, Entity3D*> mAllNoneBlockEntities;
            int mTotalDepthOfOctree;

            std::string     mMapName;

            Octree*         mRootOctree;

            // Root octree has a depth of 1, everytime it splits, the depth ++
            // So till the deepest octree every block in it is a unit block

            int             mFloorHeight; // the z of the floor
            int             mTotalUnitBlockNum;

            // it's not the boundingbox for the map, not for the octree,
            // an octree boundingbox is usually a cube, but the map is not necessary to be a cube
            AxisAlignedBox mMapBoundingBox;
        };

    }
}

#endif // _SPATIAL_OCTREE3DMAPMANAGER_H
