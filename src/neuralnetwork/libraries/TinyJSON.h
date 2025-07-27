#pragma once
// Licensed to Florent Guelfucci under one or more agreements.
// Florent Guelfucci licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
#ifndef TJ_INCLUDED 
#define TJ_INCLUDED

// Assume that we do not want std::string
#ifndef TJ_INCLUDE_STD_STRING
#define TJ_INCLUDE_STD_STRING 0
#endif

// use the std vector or not, (use the custom array).
// using the vector can cause performance issue as the
// array is optimised for deep searches.
#ifndef TJ_INCLUDE_STDVECTOR
#define TJ_INCLUDE_STDVECTOR 0
#endif

#include <exception>
#include <vector>

#if TJ_INCLUDE_STDVECTOR == 0
#include <stdexcept>
#endif

#if TJ_INCLUDE_STD_STRING == 1
#include <string>
#endif

// https://semver.org/
// Semantic Versioning 2.0.0
//   MAJOR version when you make incompatible API changes
//   MINOR version when you add functionality in a backward compatible manner
//   PATCH version when you make backward compatible bug fixes
// v0.1.1 - added some add( ... ) and set( ... ) methods
// v0.1.2 - added a but of get/set( ... ) for values and objects.
static const short TJ_VERSION_MAJOR = 0;
static const short TJ_VERSION_MINOR = 1;
static const short TJ_VERSION_PATCH = 2;
static const char TJ_VERSION_STRING[] = "0.1.2";

#ifndef TJ_USE_CHAR
#  define TJ_USE_CHAR 1
#endif

#if TJ_USE_CHAR == 1
#  define TJCHAR char
#  define TJCHARPREFIX(x) x
#elif TJ_USE_CHAR == 8
#  define TJCHAR char8_t
#  define TJCHARPREFIX(x) u8 ## x
#elif TJ_USE_CHAR == 16
#  define TJCHAR char16_t
#  define TJCHARPREFIX(x) u ## x
#elif TJ_USE_CHAR == 32
#  define TJCHAR char32_t
#  define TJCHARPREFIX(x) U ## x
#endif

namespace TinyJSON
{
#if TJ_INCLUDE_STDVECTOR == 1
#define TJDICTIONARY std::vector<TJMember*>
#define TJLIST std::vector<TJValue*>
#else
class TJList;
class TJDictionary;
#define TJDICTIONARY TJDictionary
#define TJLIST TJList
#endif

  // the various types of formating.
  enum class formating
  {
    minify,
    indented
  };

  /// <summary>
  /// The parsing options.
  /// </summary>
  struct parse_options
  {
    enum specification
    {
      rfc4627,
      rfc7159,
      rfc8259
    };

    /// <summary>
    /// The RFC specification we want to follow.
    /// </summary>
    specification specification = rfc8259;

    /// <summary>
    /// If we want to throw an exception or not.
    /// </summary>
    bool throw_exception = false;

    /// <summary>
    /// How deep we want to allow the array/objects to recurse.
    /// </summary>
    unsigned int max_depth = 64;  
  };

  /// <summary>
  /// The write options.
  /// </summary>
  struct write_options
  {
    enum byte_order_mark
    {
      none,
      utf8
    };

    /// <summary>
    /// If we want to throw an exception or not.
    /// </summary>
    bool throw_exception = false;

    /// <summary>
    /// The formating type we want to use.
    /// </summary>
    formating write_formating = formating::indented;

    /// <summary>
    /// The byte order mark we will be using.
    /// </summary>
    byte_order_mark byte_order_mark = none;
  };

  class TJWriteException : public std::exception
  {
  public:
    TJWriteException(const char* message);
    TJWriteException(const TJWriteException& exception);
    TJWriteException& operator=(const TJWriteException& exception);
    virtual ~TJWriteException() noexcept;

    virtual const char* what() const noexcept;

  private:
    void free_message() noexcept;
    void assign_message(const char* message);
    char* _message;
  };

  class TJParseException : public std::exception
  {
  public:
    TJParseException(const char* message);
    TJParseException(const TJParseException& exception);
    TJParseException& operator=(const TJParseException& exception);
    virtual ~TJParseException() noexcept;

    virtual const char* what() const noexcept;

  private:
    void free_message() noexcept;
    void assign_message(const char* message);
    char* _message;
  };

  struct internal_dump_configuration;
  class TJHelper;
  class TJValueArray;
  class TJValueObject;

  // A simple JSON value, the base of all items in a json
  class TJValue
  {
    friend TJValueArray;
    friend TJValueObject;
  public:
    TJValue();
    virtual ~TJValue();

    virtual bool is_object() const;
    virtual bool is_array() const;
    virtual bool is_string() const;
    virtual bool is_number() const;
    virtual bool is_true() const;
    virtual bool is_false() const;
    virtual bool is_null() const;

    const TJCHAR* dump(formating formating = formating::indented, const TJCHAR* indent = TJCHARPREFIX("  ")) const;
    const TJCHAR* dump_string() const;

    /// <summary>
    /// Allow each derived class to create a copy of itself.
    /// </summary>
    /// <returns></returns>
    TJValue* clone() const;

    bool get_boolean(bool strict = false) const;
    long double get_float(bool strict = false) const;
    long long get_number(bool strict = false) const;
    const TJCHAR* get_string(bool strict = false) const;
    std::vector<long double> get_floats(bool strict = false) const;
    std::vector<long long> get_numbers(bool strict = false) const;

  protected:
    /// <summary>
    /// Allow each derived class to create a copy of itself.
    /// </summary>
    /// <returns></returns>
    virtual TJValue* internal_clone() const = 0;

    virtual void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const = 0;

  private:
    TJValue(const TJValue&) = delete;
    TJValue(TJValue&&) = delete;
    TJValue& operator=(TJValue&&) = delete;
    TJValue& operator=(const TJValue&) = delete;

    mutable TJCHAR* _last_dump;
    void free_last_dump() const;
  };

  // The parser class
  class TJ
  {
  public:
    virtual ~TJ() = default;

    /// <summary>
    /// Return if the given source is valid or not.
    /// </summary>
    /// <param name="source"></param>
    /// <param name="parse_options"></param>
    /// <returns></returns>
    static bool is_valid(const TJCHAR* source, const parse_options& parse_options = {});

    /// <summary>
    /// Parse a json string
    /// </summary>
    /// <param name="source">The source we are trying to parse.</param>
    /// <param name="parse_options">The option we want to use when parsing this.</param>
    /// <returns></returns>
    static TJValue* parse(const TJCHAR* source, const parse_options& parse_options = {});

    /// <summary>
    /// Parse a json file
    /// </summary>
    /// <param name="file_path">The source file we are trying to parse.</param>
    /// <param name="parse_options">The option we want to use when parsing this.</param>
    /// <returns></returns>
    static TJValue* parse_file(const TJCHAR* file_path, const parse_options& parse_options = {});

    /// <summary>
    /// Write a value to a file.
    /// </summary>
    /// <param name="file_path">The path of the file.</param>
    /// <param name="root">the value we are writting</param>
    /// <param name="write_options">The options we will be using to write</param>
    /// <returns></returns>
    static bool write_file(const TJCHAR* file_path, const TJValue& root, const write_options& write_options = {});

  protected:
    /// <summary>
    /// Internal parsing of a json source
    /// We will use the option to throw, (or not).
    /// </summary>
    /// <param name="source"></param>
    /// <param name="parse_options"></param>
    /// <returns></returns>
    static TJValue* internal_parse(const TJCHAR* source, const parse_options& parse_options);

    /// <summary>
    /// Write a value to a file.
    /// </summary>
    /// <param name="file_path">The path of the file.</param>
    /// <param name="root">the value we are writting</param>
    /// <param name="write_options">The options we will be using to write</param>
    /// <returns></returns>
    static bool internal_write_file(const TJCHAR* file_path, const TJValue& root, const write_options& write_options);

  private:
    TJ() = delete;
    TJ(TJ&&) = delete;
    TJ(const TJ&) = delete;
    TJ& operator=(const TJ&) = delete;
    TJ& operator=(TJ&&) = delete;
  };

  // A TJMember is a key value pair, (name/value), that belong to an object.
  class TJMember
  {
    friend TJHelper;
    friend TJValueObject;
  public:
    TJMember(const TJCHAR* string, const TJValue* value);
    virtual ~TJMember();

    const TJCHAR* name() const;
    const TJValue* value() const;

  protected:
    /// <summary>
    /// Move the value ownership to the member.
    /// The caller will _not_ delete!
    /// </summary>
    /// <param name="string"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    static TJMember* move(TJCHAR*& string, TJValue*& value);

    /// <summary>
    /// Move a value to the member
    /// </summary>
    void move_value(TJValue*& value);

  private:
    TJCHAR* _string;
    TJValue* _value;
    void free_string();
    void free_value();
  };

  // A Json object that contain an array of key/value pairs.
  class TJValueObject : public TJValue
  {
    friend TJHelper;
  public:
    TJValueObject();
    virtual ~TJValueObject();

    /// <summary>
    /// Get the number of items in this array
    /// </summary>
    /// <returns></returns>
    unsigned int get_number_of_items() const;

    /// <summary>
    /// Try and get a string value, if it does not exist, then we return null.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    const TJCHAR* try_get_string(const TJCHAR* key, bool case_sensitive = true) const;

#if TJ_INCLUDE_STD_STRING == 1
    /// <summary>
    /// Try and get a string value, if it does not exist, then we return null.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    inline const TJCHAR* try_get_string(const std::string& key, bool case_sensitive = true) const
    {
      return try_get_string(key.c_str(), case_sensitive);
    }
#endif

    /// <summary>
    /// Try and get the value of this member if it exists.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    virtual const TJValue* try_get_value(const TJCHAR* key, bool case_sensitive = true) const;

    bool get_boolean(const TJCHAR* key, bool case_sensitive = true, bool throw_if_not_found = false) const;
    long double get_float(const TJCHAR* key, bool case_sensitive = true, bool throw_if_not_found = false) const;
    long long get_number(const TJCHAR* key, bool case_sensitive = true, bool throw_if_not_found = false) const;
    const TJCHAR* get_string(const TJCHAR* key, bool case_sensitive = true, bool throw_if_not_found = false) const;
    std::vector<long double> get_floats(const TJCHAR* key, bool case_sensitive = true, bool throw_if_not_found = false) const;
    std::vector<long long> get_numbers(const TJCHAR* key, bool case_sensitive = true, bool throw_if_not_found = false) const;

    void set_floats(const TJCHAR* key, const std::vector<long double>& values);
    void set_numbers(const TJCHAR* key, const std::vector<long long>& values);
    
#if TJ_INCLUDE_STD_STRING == 1
    inline bool get_boolean(const std::string& key, bool case_sensitive = true, bool throw_if_not_found = false) const
    {
      return get_boolean(key.c_str(), case_sensitive, throw_if_not_found);
    }
    inline long double get_float(const std::string& key, bool case_sensitive = true, bool throw_if_not_found = false) const
    {
      return get_float(key.c_str(), case_sensitive, throw_if_not_found);
    }
    inline long long get_number(const std::string& key, bool case_sensitive = true, bool throw_if_not_found = false) const
    {
      return get_number(key.c_str(), case_sensitive, throw_if_not_found);
    }
    inline const TJCHAR* get_string(const std::string& key, bool case_sensitive = true, bool throw_if_not_found = false) const
    {
      return get_string(key.c_str(), case_sensitive, throw_if_not_found);
    }
    inline std::vector<long double> get_floats(const std::string& key,bool case_sensitive = true, bool throw_if_not_found = false) const
    {
      return get_floats(key.c_str(), case_sensitive, throw_if_not_found);
    }
    inline std::vector<long long> get_numbers(const std::string& key, bool case_sensitive = true, bool throw_if_not_found = false) const
    {
      return get_numbers(key.c_str(), case_sensitive, throw_if_not_found);
    }

    /// <summary>
    /// Try and get the value of this member if it exists.
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    inline const TJValue* try_get_value(const std::string& key, bool case_sensitive = true) const
    {
      return try_get_value(key.c_str(), case_sensitive);
    }
#endif

    TJMember* operator [](int idx) const;
    TJMember* at(int idx) const;

    bool is_object() const override;

    /// <summary>
    /// Set the value of a ... value
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    void set(const TJCHAR* key, const TJValue* value);

    /// <summary>
    /// Set the value of a ... value
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    inline void set(const TJCHAR* key, const TJValue& value)
    {
      set(key, &value);
    }

    /// <summary>
    /// Set the value of a number
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    void set_number(const TJCHAR* key, long long value);

    /// <summary>
    /// Set the value of a number
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    void set_float(const TJCHAR* key, long double value);

    /// <summary>
    /// Set the value a boolean
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    void set_boolean(const TJCHAR* key, bool value);

    /// <summary>
    /// Set the value of a string.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    void set_string(const TJCHAR* key, const char* value);

    /// <summary>
    /// Set the value to null.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    void set_null(const TJCHAR* key);

#if TJ_INCLUDE_STD_STRING == 1
    /// <summary>
    /// Set the value of a string.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    inline void set_string(const std::string& key, const std::string& value) const
    {
      return set_string(key.c_str(), value.c_str());
    }
#endif

    /// <summary>
    /// Pop a value out of the list of items
    /// </summary>
    /// <param name="key"></param>
    void pop(const TJCHAR* key);

#if TJ_INCLUDE_STD_STRING == 1
    /// <summary>
    /// Pop a value out of the list of items
    /// </summary>
    /// <param name="key"></param>
    inline void pop(const std::string& key)
    {
      pop(key.c_str());
    }
#endif
  protected:
    /// <summary>
    /// Clone an array into an identical array
    /// </summary>
    TJValue* internal_clone() const override;

    /// <summary>
    /// Move the value ownership to the members.
    /// The caller will _not_ delete!
    /// </summary>
    /// <param name="members"></param>
    /// <returns></returns>
    static TJValueObject* move(TJDICTIONARY*& members);

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;

  private:
    // All the key value pairs in this object.
    TJDICTIONARY* _members;

    void free_members();
  };

  // A Json object that contain an array of key/value pairs.
  class TJValueArray : public TJValue
  {
    friend TJHelper;
  public:
    TJValueArray();
    virtual ~TJValueArray();

    /// <summary>
    /// Get the number of items in this array
    /// </summary>
    /// <returns></returns>
    unsigned int get_number_of_items() const;

    TJValue* operator [](int idx) const;
    TJValue* at(int idx) const;

    bool is_array() const override;

    void add(const TJValue* value);
    void add_numbers(const std::vector<long long>& values);
    void add_numbers(const std::vector<long>& values);
    void add_numbers(const std::vector<int>& values);
    void add_floats(const std::vector<long double>& values);
    void add_floats(const std::vector<double>& values);
    void add_floats(const std::vector<float>& values);
    void add_number(long long value);
    void add_float(long double value);
    void add_boolean(bool value);
    void add_string(const char* value);

    std::vector<long double> get_floats(bool throw_if_not_numbers = false) const;
    std::vector<long long> get_numbers(bool throw_if_not_numbers = false) const;

#if TJ_INCLUDE_STD_STRING == 1
    void add_string(const std::string& value)
    {
      add_string(value.c_str());
    }
#endif

  protected:
    /// <summary>
    /// Clone an array into an identical array
    /// </summary>
    TJValue* internal_clone() const override;

    /// <summary>
    /// Move the value ownership to the values.
    /// The caller will _not_ delete!
    /// </summary>
    /// <param name="values"></param>
    /// <returns></returns>
    static TJValueArray* move(TJLIST*& values);

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;
    
  private:
    // All the key value pairs in this object.
    TJLIST* _values;

    void free_values();
  };

  // A string JSon
  class TJValueString : public TJValue
  {
    friend TJHelper;
  public:
    TJValueString(const TJCHAR* value);
    virtual ~TJValueString();

    bool is_string() const override;

    const TJCHAR* raw_value() const;

  protected:
    /// <summary>
    /// Clone an array into an identical array
    /// </summary>
    TJValue* internal_clone() const override;

    /// <summary>
    /// Move the value ownership of the string.
    /// The caller will _not_ delete!
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    static TJValueString* move(TJCHAR*& value);

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;

  private:
    TJCHAR* _value;
    void free_value();
  };

  // A bolean JSon
  class TJValueBoolean : public TJValue
  {
  public:
    TJValueBoolean(bool is_true);
    virtual ~TJValueBoolean() = default;

    bool is_true() const override;
    bool is_false() const override;

  protected:
    /// <summary>
    /// Clone an array into an identical array
    /// </summary>
    TJValue* internal_clone() const override;

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;

  private:
    const bool _is_true;
  };

  // A null JSon
  class TJValueNull : public TJValue
  {
  public:
    TJValueNull();
    virtual ~TJValueNull() = default;

    bool is_null() const override;

  protected:
    /// <summary>
    /// Clone an array into an identical array
    /// </summary>
    TJValue* internal_clone() const override;

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;
  };

  // A number JSon, float or int
  class TJValueNumber : public TJValue
  {
  protected:
    TJValueNumber(const bool is_negative);
    virtual ~TJValueNumber() = default;

  public:
    bool is_number() const override;

    long double get_float() const;
    long long get_number() const;

  protected:
    const bool _is_negative;
  };

  // A number JSon, float or int
  class TJValueNumberInt : public TJValueNumber
  {
  public:
    TJValueNumberInt(const long long& number);
    TJValueNumberInt(const unsigned long long& number, const bool is_negative);
    virtual ~TJValueNumberInt() = default;

    long long get_number() const;

  protected:
    /// <summary>
    /// Clone an array into an identical array
    /// </summary>
    TJValue* internal_clone() const override;

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;

  private:
    const long long _number;
  };

  // A number JSon, float or int
  class TJValueNumberFloat : public TJValueNumber
  {
  public:
    TJValueNumberFloat(long double number);
    TJValueNumberFloat(const unsigned long long& number, const unsigned long long& fraction, const unsigned int& fraction_exponent, bool is_negative);
    virtual ~TJValueNumberFloat();

    long double get_number() const;

  protected:
    /// <summary>
    /// Clone an array into an identical array
    /// </summary>
    TJValue* internal_clone() const override;

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;

  private:
  private:
    void make_string_if_needed() const;
    mutable TJCHAR* _string;
    const unsigned long long _number;
    const unsigned long long _fraction;
    const unsigned int _fraction_exponent;
  };

  // A number JSon, float or int
  class TJValueNumberExponent : public TJValueNumber
  {
  public:
    TJValueNumberExponent(const unsigned long long& number, const unsigned long long& fraction, const unsigned int& fraction_exponent, const int& exponent,bool is_negative);
    virtual ~TJValueNumberExponent();

    long double get_number() const;

  protected:
    /// <summary>
    /// Clone an array into an identical value object
    /// </summary>
    TJValue* internal_clone() const override;

    void internal_dump(internal_dump_configuration& configuration, const TJCHAR* current_indent) const override;

  private:
    void make_string_if_needed() const;
    mutable TJCHAR* _string;
    const unsigned long long _number;
    const unsigned long long _fraction;
    const unsigned int _fraction_exponent;
    const int _exponent;
  };


  // user_literals
  inline TJValue* operator ""_tj(const TJCHAR * source, std::size_t)
  {
    parse_options options = {};
    options.throw_exception = true;
    return TJ::parse(source, options);
  }

  #if TJ_INCLUDE_STD_STRING == 1
  inline std::string operator ""_tj_indent(const TJCHAR * source, std::size_t)
  {
    parse_options options = {};
    options.throw_exception = true;
    auto* tj = TJ::parse(source, options);
    if (nullptr == tj)
    {
      //  exception will throw.
      return TJCHARPREFIX("");
    }
    std::string json(tj->dump(formating::indented));
    delete tj;
    return json;
  }  

  inline std::string operator ""_tj_minify(const TJCHAR * source, std::size_t)
  {
    parse_options options = {};
    options.throw_exception = true;
    auto* tj = TJ::parse(source, options);
    if (nullptr == tj)
    {
      //  exception will throw.
      return TJCHARPREFIX("");
    }
    std::string json(tj->dump(formating::minify));
    delete tj;
    return json;
  }
  #endif
} // TinyJSON
#endif // !TJ_INCLUDED 