#pragma once

#include <cstdint>
#include "interfaces/ICommLink.hpp"
#include "interfaces/IGoal.hpp"
#include "interfaces/IOffboardApi.hpp"
#include "interfaces/IUpdatable.hpp"
#include "interfaces/CommonStructs.hpp"
#include "RemoteControl.hpp"
#include "Params.hpp"

namespace simple_flight {

class OffboardApi : 
    public IUpdatable,
    public IOffboardApi {
public:
    OffboardApi(const Params* params, const IBoardClock* clock, const IBoardInputPins* board_inputs, 
        IStateEstimator* state_estimator, ICommLink* comm_link)
        : params_(params), rc_(params, clock, board_inputs, &vehicle_state_, state_estimator, comm_link), 
          state_estimator_(state_estimator), comm_link_(comm_link), clock_(clock)
    {
    }

    virtual void reset() override
    {
        IUpdatable::reset();

        vehicle_state_.setState(params_->default_vehicle_state, state_estimator_->getGeoPoint());
        rc_.reset();
        has_api_control_ = false;
        goal_timestamp_ = 0;
        updateGoalFromRc();
    }

    virtual void update() override
    {
        IUpdatable::update();

        rc_.update();
        if (!has_api_control_)
            updateGoalFromRc();
        else {
            if (clock_->millis() - goal_timestamp_ > params_->api_goal_timeout) {
                if (!is_api_timedout_) {
                    /*
                    comm_link_->log("API call timed out, entering hover mode");
                    goal_mode_ = GoalMode::getPositionMode();
                    goal_ = Axis4r::xyzToAxis4(state_estimator_->getPosition(), true);
                    */
                    is_api_timedout_ = true;
                }

                //do not update goal_timestamp_
            }

        }
        //else leave the goal set by IOffboardApi API
    }

    /**************** IOffboardApi ********************/

    virtual const Axis4r& getGoalValue() const override
    {
        return goal_;
    }

    virtual const GoalMode& getGoalMode() const override
    {
        return goal_mode_;
    }


    virtual bool canRequestApiControl(std::string& message) override
    {
        if (rc_.allowApiControl())
            return true;
        else {
            message = "Remote Control switch position disallows API control";
            comm_link_->log(message, ICommLink::kLogLevelError);
            return false;
        }
    }
    virtual bool hasApiControl() override
    {
        return has_api_control_;
    }
    virtual bool requestApiControl(std::string& message) override
    {
        if (canRequestApiControl(message)) {
            has_api_control_ = true;

            //initial value from RC for smooth transition
            updateGoalFromRc();

            comm_link_->log("requestApiControl was successful", ICommLink::kLogLevelInfo);

            return true;
        } else {
            comm_link_->log("requestApiControl failed", ICommLink::kLogLevelError);
            return false;
        }
    }
    virtual void releaseApiControl() override
    {
        has_api_control_ = false;
        comm_link_->log("releaseApiControl was sucessful", ICommLink::kLogLevelInfo);
    }
    virtual bool setGoalAndMode(const Axis4r* goal, const GoalMode* goal_mode, std::string& message) override
    {
        if (has_api_control_) {
            if (goal != nullptr)
                goal_ = *goal;
            if (goal_mode != nullptr)
                goal_mode_ = *goal_mode;
            goal_timestamp_ = clock_->millis();
            is_api_timedout_ = false;
            return true;
        } else {
            message = "requestApiControl() must be called before using API control";
            comm_link_->log(message, ICommLink::kLogLevelError);
            return false;
        }

    }

    virtual bool arm(std::string& message) override
    {
        if (has_api_control_) {
            if (vehicle_state_.getState() == VehicleStateType::Armed) {
                message = "Vehicle is already armed";
                comm_link_->log(message, ICommLink::kLogLevelInfo);
                return true;
            }
            else if ((vehicle_state_.getState() == VehicleStateType::Inactive
                || vehicle_state_.getState() == VehicleStateType::Disarmed
                || vehicle_state_.getState() == VehicleStateType::BeingDisarmed)) {

                state_estimator_->setHomeGeoPoint(state_estimator_->getGeoPoint());
                vehicle_state_.setState(VehicleStateType::Armed, state_estimator_->getHomeGeoPoint());
                goal_ = Axis4r(0, 0, 0, params_->rc.min_angling_throttle);
                goal_mode_ = GoalMode::getAllRateMode();

                message = "Vehicle is armed";
                comm_link_->log(message, ICommLink::kLogLevelInfo);
                return true;
            }
            else {
                message = "Vehicle cannot be armed because it is not in Inactive, Disarmed or BeingDisarmed state";
                comm_link_->log(message, ICommLink::kLogLevelError);
                return false;
            }
        }
        else {
            message = "Vehicle cannot be armed via API because API has not been given control";
            comm_link_->log(message, ICommLink::kLogLevelError);
            return false;
        }

    }

    virtual bool disarm(std::string& message) override
    {
        if (has_api_control_ && (vehicle_state_.getState() == VehicleStateType::Active
            || vehicle_state_.getState() == VehicleStateType::Armed
            || vehicle_state_.getState() == VehicleStateType::BeingArmed)) {

            vehicle_state_.setState(VehicleStateType::Disarmed);
            goal_ = Axis4r(0, 0, 0, 0);
            goal_mode_ = GoalMode::getAllRateMode();

            message = "Vehicle is disarmed";
            comm_link_->log(message, ICommLink::kLogLevelInfo);
            return true;
        }
        else {
            message = "Vehicle cannot be disarmed because it is not in Active, Armed or BeingArmed state";
            comm_link_->log(message, ICommLink::kLogLevelError);
            return false;
        }
    }

    virtual VehicleStateType getVehicleState() const override
    {
        return vehicle_state_.getState();
    }

    virtual const IStateEstimator& getStateEstimator() override
    {
        return *state_estimator_;
    }

    virtual GeoPoint getHomeGeoPoint() const override
    {
        return state_estimator_->getHomeGeoPoint();
    }
    
    virtual GeoPoint getGeoPoint() const override
    {
        return state_estimator_->getGeoPoint();
    }


private:
    void updateGoalFromRc()
    {
        goal_ = rc_.getGoalValue();
        goal_mode_ = rc_.getGoalMode();
    }

private:
    const Params* params_;
    RemoteControl rc_;
    IStateEstimator* state_estimator_;
    ICommLink* comm_link_;
    const IBoardClock* clock_;

    VehicleState vehicle_state_;
    
    Axis4r goal_;
    GoalMode goal_mode_;
    uint64_t goal_timestamp_;

    bool has_api_control_;
    bool is_api_timedout_;
};


} //namespace