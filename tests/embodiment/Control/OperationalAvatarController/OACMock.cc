/*
 * tests/embodiment/Control/OperationalAvatarController/OACMock.cc
 *
 * @author Zhenhua Cai <czhedu@gmail.com>
 * @date   2011-06-14
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "OACMock.h"

void OACMock::setConfig() {
    // Set arguments used by OAC::init
    // You can easily get these arguments by set MANUAL_OAC_LAUNCH true in 
    // config file, run the embodiment system and copy the command of 
    // running OAC in console. 
    agentBrainId = "123456"; 
    ownerId = "290"; 
    agentType = "pet";  
    agentTrait = "Maxie"; 
    networkElementPort = "16326"; 
    cogServerShellPort = "17002"; 
    zmqPublishPort = "18002"; 

    // Set logger level
    logger().setLevel(Logger::FINE);
    logger().setPrintToStdoutFlag(true);

    // Load config file and reset some configurations  
    config(opencog::control::EmbodimentConfig::embodimentCreateInstance, true);
    
    if ( fileExists(config().get("CONFIG_FILE").c_str()) ) {
        config().load(config().get("CONFIG_FILE").c_str());
    }
   
    config().set("EXTERNAL_TICK_MODE", "true");
    config().set("SERVER_PORT", cogServerShellPort);
    config().set("ZMQ_PUBLISH_PORT", zmqPublishPort);

    // Reset module list to skip PLN module. Because BackChainingAgent and
    // ForwardChainingAgent in PLN module will make OAC crash in unit test when
    // unload the PLN module. In addition, it seems these two agents are never 
    // started.  
    config().set("MODULES", "libquery.so, libbuiltinreqs.so, \
                             libscheme-shell.so, libpersist.so");

    config().set("RUN_OAC_DEBUGGER", "false"); 
    config().set("MANUAL_OAC_LAUNCH", "false"); 
    config().set("ENABLE_UNITY_CONNECTOR", "false"); 

    // OAC fails to create the log file in /tmp/... during unit test (why?), so
    // we change the log file to user's home directory and also print log to 
    // the screen. 
    config().set("LOG_DIR", "/home/$USER"); 
    config().set("PRINT_LOG_TO_STDOUT", "true");

    // Disable opencog::spatial::MapExplorerServer, which raises
    // 'bad file descriptor' error during unit test
    config().set("VISUAL_DEBUGGER_ACTIVE", "false"); 
};

OAC & OACMock::createOAC()
{
    // Create an instance of OAC
    server(OAC::createInstance);
    this->oac = & static_cast<OAC&>(server());

    // Open database *before* loading modules, since the modules
    // might create atoms, and we can't have that happen until 
    // storage is open, as otherwise, there will be handle conflicts.
    oac->openDatabase();

    // Load modules specified in config
    oac->loadModules(); 

    const char* config_path[] = {"."};
    oac->loadSCMModules(config_path);

    // Initialize OAC
    //
    // OAC::loadSCMModules should be called before calling OAC::init, 
    // because OAC::loadSCMModules will load 'rules_core.scm',  which should be loaded 
    // before loading Psi Rules ('xxx_rules.scm') and 
    // OAC::init is responsible for loading Psi Rules via OAC::addRulesToAtomSpace
    int portNumber = boost::lexical_cast<int>(networkElementPort); 
    oac->init(
        agentBrainId,       // agent-brain-id, i.e., id of OAC
        "127.0.0.1",        // NetworkElement ip 
        portNumber,         // NetworkElement port number
        zmqPublishPort,     // ZeroMQ port used by subscribers to get messages
        PAIUtils::getInternalId(agentBrainId.c_str()), // pet id 
        PAIUtils::getInternalId(ownerId.c_str()),      // owner id 
        agentType,  // agent type
        agentTrait  // agent traits
    );

    // enable the network server
    oac->enableNetworkServer();

    // Return the reference of newly created OAC
    return *(this->oac); 
}

Message * OACMock::createMessageFromFile(const std::string & from, 
                                         const std::string & to, 
                                         int msgType,
                                         const char * fileName)
{
    Message * p_message = NULL; 

    std::ifstream in(fileName);

    if ( in.good() ) {
        std::istreambuf_iterator<char> beg(in), end; 
        std::string msgContent(beg, end); 
        p_message = Message::factory(from, to, msgType, msgContent);
    }
    else {
        logger().error("OACMock::%s - Fail to create message from file '%s'", 
                       __FUNCTION__, 
                       fileName
                      ); 
    }

    in.close(); 

    return p_message; 
}

